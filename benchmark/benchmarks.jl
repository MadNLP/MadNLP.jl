using Pkg, Distributed

addprocs(4,exeflags="--project=.")
Pkg.instantiate()

@everywhere using CUTEst, JLD2

if ARGS[1] == "master"

    Pkg.add(PackageSpec(name="MadNLP",rev="master"))
    Pkg.add(PackageSpec(name="MadNLPHSL",rev="master"))
    Pkg.build("MadNLPHSL")
    @everywhere solver = nlp -> madnlp(nlp,linear_solver=MadNLPMa57,tol=1e-6)
    @everywhere using MadNLP, MadNLPHSL

elseif ARGS[1] == "current"

    Pkg.develop(path=joinpath(@__DIR__,".."))
    Pkg.develop(path=joinpath(@__DIR__,"..","lib","MadNLPHSL"))
    Pkg.build("MadNLPHSL")
    @everywhere solver = nlp -> madnlp(nlp,print_level=MadNLP.DEBUG,linear_solver=MadNLPMa57,tol=1e-6)
    @everywhere using MadNLP, MadNLPHSL

elseif ARGS[1] == "ipopt"

    @everywhere solver = nlp -> ipopt(nlp,linear_solver="ma57",tol=1e-6)
    @everywhere using NLPModelsIpopt

elseif ARGS[1] == "knitro"
    # TODO
else
    error("Proper ARGS[1] should be given")
end


function decodemodel(name)
    finalize(CUTEstModel(name))
end

@everywhere function evalmodel(name,solver)
    println(name)
    nlp = CUTEstModel(name; decode=false)
    retval = solver(nlp)
    finalize(nlp)
    retval
end

function benchmark(solver,probs)
    println("Decoding problems")
    decodemodel.(probs)
    println("Solving problems")
    retvals = pmap(prob->evalmodel(prob,solver),probs)
    time   = [retval.elapsed_time for retval in retvals]
    status = [retval.status       for retval in retvals]
    time,status
end

function benchmark1(solver,prob)
    println("Decoding problems")
    decodemodel(prob)
    println("Solving problems")
    retval = evalmodel(prob,solver)
    retval.elapsed_time, retval.status
end


exclude = [
    "PFIT1","PFIT2","PFIT4", # madnlp fails
    "DENSCHNE","SPECANNE","DJTL", #ipopt fails
    "PRIMAL3", # madnlp fails
    "TAX213322","TAXR213322", "HIMMELP2","MOSARQP2"
    # "CVXQP2", "TAX213322", "OSORIO", "BIGGSB1", # almost converged
]

# probs = CUTEst.select()
probs = CUTEst.select(max_var=5000)
# probs = CUTEst.select(min_var=1501,max_var=2000) #- 3
# probs = CUTEst.select(min_var=2001,max_var=5000) #- 4
# probs = CUTEst.select(min_var=5001,max_var=10000) #- 2
# probs = CUTEst.select(min_var=10001,max_var=20000) #- 0
# probs = CUTEst.select(min_var=20001,max_var=Inf) #- 5

filter!(e->!(e in exclude),probs)

time,status = benchmark(solver,probs)
# save_object("time-$(ARGS[1]).jld2",time)
# save_object("status-$(ARGS[1]).jld2",status) 








