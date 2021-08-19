include("config.jl")

function get_status(code::Symbol) 
    if code == :first_order
        return 1
    elseif code == :acceptable
        return 2
    else
        return 3
    end
end

@everywhere using CUTEst

if SOLVER == "master"
    @everywhere solver = nlp -> madnlp(nlp,linear_solver=MadNLPMa57,max_wall_time=900.,tol=1e-6)
    @everywhere using MadNLP, MadNLPHSL
elseif SOLVER == "current"
    @everywhere solver = nlp -> madnlp(nlp,linear_solver=MadNLPMa57,max_wall_time=900.,tol=1e-6)
    @everywhere using MadNLP, MadNLPHSL
elseif SOLVER == "ipopt"
    @everywhere solver = nlp -> ipopt(nlp,linear_solver="ma57",max_cpu_time=900.,tol=1e-6)
    @everywhere using NLPModelsIpopt
elseif SOLVER == "knitro"
    # TODO
else
    error("Proper SOLVER should be given")
end


@everywhere function decodemodel(name)
    println("Decoding $name")
    finalize(CUTEstModel(name))
end

@everywhere function evalmodel(name,solver)
    println("Solving $name")
    nlp = CUTEstModel(name; decode=false)
    t = @elapsed begin
        retval = solver(nlp)
    end
    retval.elapsed_time = t
    finalize(nlp)
    return retval
end

function benchmark(solver,probs;warm_up_probs = [])
    println("Warming up (forcing JIT compile)")
    broadcast(decodemodel,warm_up_probs)
    [remotecall_fetch.(prob->evalmodel(prob,solver),i,warm_up_probs) for i in procs() if i!= 1]

    println("Decoding problems")
    broadcast(decodemodel,probs)
    
    println("Solving problems")
    retvals = pmap(prob->evalmodel(prob,solver),probs)
    time   = [retval.elapsed_time for retval in retvals]
    status = [get_status(retval.status) for retval in retvals]
    time,status
end

exclude = [
    "PFIT1","PFIT2","PFIT4","DENSCHNE","SPECANNE","DJTL", "EG3","OET7",
    "PRIMAL3","TAX213322","TAXR213322","TAX53322","TAXR53322","HIMMELP2","MOSARQP2","LUKVLE11",
    "CYCLOOCT","CYCLOOCF","LIPPERT1","GAUSSELM","A2NSSSSL",
    "YATP1LS","YATP2LS","YATP1CLS","YATP2CLS","BA-L52LS","BA-L73LS","BA-L21LS","CRESC132"    
]


probs = CUTEst.select()

println(probs)
filter!(e->!(e in exclude),probs)

time,status = benchmark(solver,probs;warm_up_probs = ["EIGMINA"])

writedlm("name-cutest.csv",probs,',')
writedlm("time-cutest-$(SOLVER).csv",time,',')
writedlm("status-cutest-$(SOLVER).csv",status),',' 
