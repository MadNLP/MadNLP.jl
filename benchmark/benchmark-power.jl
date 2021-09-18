include("config.jl")

@everywhere begin
    if haskey(ENV, "PGLIB_PATH")
        const PGLIB_PATH = ENV["PGLIB_PATH"]
    else
        error("Unable to find path to PGLIB benchmark.\n"*
            "Please set environment variable `PGLIB_PATH` to run benchmark with PowerModels.jl")
    end

    using PowerModels, MathOptInterface, JuMP
    const MOI = MathOptInterface
    
    PowerModels.silence()

    function evalmodel(prob,solver;gcoff=false)
        case,type = prob
        pm = instantiate_model(joinpath(PGLIB_PATH,case),type,PowerModels.build_opf)
        println("Solving $(get_name(pm))")
        gcoff && GC.enable(false);
        retval = solver(pm)
        gcoff && GC.enable(true);
        return retval
    end

    function get_status(code::MOI.TerminationStatusCode)
        if code == MOI.LOCALLY_SOLVED
            return 1
        elseif code == MOI.ALMOST_OPTIMAL
            return 2
        else
            return 3
        end
    end

    get_name(pm) = "$(pm.data["name"])-$(typeof(pm))"
end

if SOLVER == "master" || SOLVER == "current"
    @everywhere begin
        using MadNLP, MadNLPHSL
        solver = pm -> begin
            set_optimizer(pm.model,()->
                MadNLP.Optimizer(linear_solver=MadNLPMa57,max_wall_time=900.,tol=1e-6, print_level=PRINT_LEVEL))
            mem=@allocated begin
                t=@elapsed begin
                    optimize_model!(pm)
                end
            end
            return get_status(termination_status(pm.model)),t,mem,barrier_iterations(pm.model)
        end
    end
elseif SOLVER == "ipopt"
    @everywhere begin
        using Ipopt
        
        const ITER = [-1]
        function ipopt_callback(
            prob::IpoptProblem,alg_mod::Cint,iter_count::Cint,obj_value::Float64,
            inf_pr::Float64,inf_du::Float64,mu::Float64,d_norm::Float64,
            regularization_size::Float64,alpha_du::Float64,alpha_pr::Float64,ls_trials::Cint)
            
            ITER[] += 1
            return true
        end

        solver = pm -> begin
            ITER[] = 0
            set_optimizer(pm.model,()->
                Ipopt.Optimizer(linear_solver="ma57",max_cpu_time=900.,tol=1e-6, print_level=PRINT_LEVEL))
            MOI.set(pm.model, Ipopt.CallbackFunction(), ipopt_callback)
            mem=@allocated begin
                t=@elapsed begin
                    optimize_model!(pm)
                end
            end
            return get_status(termination_status(pm.model)),t,mem,ITER[]
        end
    end
elseif SOLVER == "knitro"
    # TODO
else
    error("Proper SOLVER should be given")
end


function benchmark(solver,probs;warm_up_probs = [])
    println("Warming up (forcing JIT compile)")
    warm_up_pms = [
        instantiate_model(joinpath(PGLIB_PATH,case),type,PowerModels.build_opf)
        for (case,type) in warm_up_probs]
    println(get_name.(warm_up_pms))
    rs = [remotecall.(solver,i,warm_up_pms) for i in procs() if i!= 1]
    ws = [wait.(r) for r in rs]
    fs= [fetch.(r) for r in rs]

    println("Solving problems")
    retvals = pmap(prob->evalmodel(prob,solver;gcoff=GCOFF),probs)
    
    status = [status for (status,time,mem,iter) in retvals]
    time   = [time for (status,time,mem,iter) in retvals]
    mem    = [mem for (status,time,mem,iter) in retvals]
    iter   = [iter for (status,time,mem,iter) in retvals]
        
    return status,time,mem,iter
end

if QUICK
    cases = filter!(e->(occursin("pglib_opf_case",e) && occursin("pegase",e)),readdir(PGLIB_PATH))
    types = [ACPPowerModel, ACRPowerModel]
else
    cases = filter!(e->occursin("pglib_opf_case",e),readdir(PGLIB_PATH))
    types = [ACPPowerModel, ACRPowerModel, ACTPowerModel,
             DCPPowerModel, DCMPPowerModel, NFAPowerModel,
             DCPLLPowerModel,LPACCPowerModel, SOCWRPowerModel,
             QCRMPowerModel,QCLSPowerModel]
end
probs = [(case,type) for case in cases for type in types]
name =  ["$case-$type" for case in cases for type in types]

status,time,mem,iter = benchmark(solver,probs;warm_up_probs = [
    ("pglib_opf_case1888_rte.m",ACPPowerModel)
])

writedlm("name-power.csv",name,',')
writedlm("status-power-$(SOLVER).csv",status)
writedlm("time-power-$(SOLVER).csv",time)
writedlm("mem-power-$(SOLVER).csv",mem)
writedlm("iter-power-$(SOLVER).csv",iter)
