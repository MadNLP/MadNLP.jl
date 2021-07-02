const PGLIB_PATH = ENV["PGLIB_PATH"]

include("config.jl")

using MathOptInterface; const MOI=MathOptInterface
function get_status(code::MOI.TerminationStatusCode) 
    if code == MOI.LOCALLY_SOLVED
        return 1
    elseif code == MOI.ALMOST_OPTIMAL
        return 2
    else
        return 3
    end
end

@everywhere using PowerModels
@everywhere PowerModels.silence()

if SOLVER == "master"
    @everywhere solver = prob ->
        run_opf(joinpath(PGLIB_PATH,prob[1]), prob[2],
                ()->MadNLP.Optimizer(linear_solver=MadNLPMa57,max_wall_time=900.,tol=1e-6))
    @everywhere using MadNLP, MadNLPHSL
elseif SOLVER == "current"
    @everywhere solver = prob ->
        run_opf(joinpath(PGLIB_PATH,prob[1]), prob[2],
                ()->MadNLP.Optimizer(linear_solver=MadNLPMa57,max_wall_time=900.,tol=1e-6))
    @everywhere using MadNLP, MadNLPHSL
elseif SOLVER == "ipopt"
    @everywhere solver = prob ->
        run_opf(joinpath(PGLIB_PATH,prob[1]), prob[2],
                ()->Ipopt.Optimizer(linear_solver="ma57",max_cpu_time=900.,tol=1e-6))
    @everywhere using Ipopt
elseif SOLVER == "knitro"
    # TODO
else
    error("Proper SOLVER should be given")
end


@everywhere function evalmodel(prob,solver)
    println("Solving $prob")
    t = @elapsed begin
        retval = solver(prob)
    end
    retval["solve_time"] = t
    return retval
end

function benchmark(solver,probs;warm_up_probs = [])
    println("Warming up (forcing JIT compile)")
    println(warm_up_probs)
    rs = [remotecall.(solver,i,warm_up_probs) for i in procs() if i!= 1]
    ws = [wait.(r) for r in rs]
    fs= [fetch.(r) for r in rs]
    
    println("Solving problems")
    retvals = pmap(prob->evalmodel(prob,solver),probs)
    time   = [retval["solve_time"] for retval in retvals]
    status = [get_status(retval["termination_status"]) for retval in retvals]
    time,status
end

cases = filter!(e->occursin("pglib_opf_case",e),readdir(PGLIB_PATH))
types = [ACPPowerModel, ACRPowerModel, ACTPowerModel,
         DCPPowerModel, DCMPPowerModel, NFAPowerModel,
         DCPLLPowerModel,LPACCPowerModel, SOCWRPowerModel,
         QCRMPowerModel,QCLSPowerModel]
probs = [(case,type) for case in cases for type in types]
name =  ["$case-$type" for case in cases for type in types]

time,status = benchmark(solver,probs;warm_up_probs = [("pglib_opf_case1888_rte.m", ACPPowerModel)])

save_object("name-power.jld2",name)
save_object("time-power-$(SOLVER).jld2",time)
save_object("status-power-$(SOLVER).jld2",status) 
