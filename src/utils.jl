abstract type AbstractOptions end

# Build info
default_linear_solver() = UmfpackSolver
default_dense_solver() = LapackCPUSolver

# Logger
@kwdef mutable struct Logger
    print_level::LogLevels = INFO
    file_print_level::LogLevels = INFO
    file::Union{IOStream,Nothing} = nothing
end

get_level(logger::Logger) = logger.print_level
get_file_level(logger::Logger) = logger.file_print_level
get_file(logger::Logger) = logger.file
finalize(logger::Logger) = logger.file != nothing && close(logger.file)

for (name,level,color) in [(:trace,TRACE,7),(:debug,DEBUG,6),(:info,INFO,256),(:notice,NOTICE,256),(:warn,WARN,5),(:error,ERROR,9)]
    @eval begin
        macro $name(logger,str)
            gl = $get_level
            gfl= $get_file_level
            gf = $get_file
            l = $level
            c = $color
            code = quote
                if $gl($logger) <= $l
                    if $c == 256
                        println($str)
                    else
                        printstyled($str,"\n",color=$c)
                    end
                end
                if $gf($logger) != nothing && $gfl($logger) <= $l
                    println($gf($logger),$str)
                end
            end
            esc(code)
        end
    end
end

# BLAS
const blas_num_threads = Ref{Int}(1)
function set_blas_num_threads(n::Integer;permanent::Bool=false)
    permanent && (blas_num_threads[]=n)
    BLAS.set_num_threads(n)
end
macro blas_safe_threads(args...)
    code = quote
        set_blas_num_threads(1)
        Threads.@threads($(args...))
        set_blas_num_threads(blas_num_threads[])
    end
    return esc(code)
end

# unsafe wrap
function _madnlp_unsafe_wrap(vec::VT, n, shift=1) where VT
    return unsafe_wrap(VT, pointer(vec,shift), n)
end

# Type definitions for noncontiguous views
const SubVector{Tv} = SubArray{Tv, 1, Vector{Tv}, Tuple{Vector{Int}}, false}

@kwdef mutable struct Counters
    k::Int = 0 # total iteration counter
    l::Int = 0 # backtracking line search counter
    t::Int = 0 # restoration phase counter

    start_time::Float64

    linear_solver_time::Float64 = 0.
    eval_function_time::Float64 = 0.
    solver_time::Float64 = 0.
    total_time::Float64 = 0.

    obj_cnt::Int = 0
    obj_grad_cnt::Int = 0
    con_cnt::Int = 0
    con_jac_cnt::Int = 0
    lag_hess_cnt::Int = 0

    acceptable_cnt::Int = 0
end

"""
    timing_callbacks(ips::InteriorPointSolver; ntrials=10)

Return the average timings spent in each callback for `ntrials` different trials.
Results are returned inside a named-tuple.

"""
function timing_callbacks(ips; ntrials=10)
    t_f, t_c, t_g, t_j, t_h = (0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in 1:ntrials
        t_f += @elapsed eval_f_wrapper(ips, ips.x)
        t_c += @elapsed eval_cons_wrapper!(ips, ips.c, ips.x)
        t_g += @elapsed eval_grad_f_wrapper!(ips, ips.f,ips.x)
        t_j += @elapsed eval_jac_wrapper!(ips, ips.kkt, ips.x)
        t_h += @elapsed eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.l)
    end
    return (
        time_eval_objective   = t_f / ntrials,
        time_eval_constraints = t_c / ntrials,
        time_eval_gradient    = t_g / ntrials,
        time_eval_jacobian    = t_j / ntrials,
        time_eval_hessian     = t_h / ntrials,
    )
end

"""
    timing_linear_solver(ips::InteriorPointSolver; ntrials=10)

Return the average timings spent in the linear solver for `ntrials` different trials.
Results are returned inside a named-tuple.

"""
function timing_linear_solver(ips; ntrials=10)
    t_build, t_factorize, t_backsolve = (0.0, 0.0, 0.0)
    for _ in 1:ntrials
        t_build     += @elapsed build_kkt!(ips.kkt)
        t_factorize += @elapsed factorize!(ips.linear_solver)
        t_backsolve += @elapsed solve_refine_wrapper!(ips,ips.d,ips.p)
    end
    return (
        time_build_kkt = t_build / ntrials,
        time_factorization = t_factorize / ntrials,
        time_backsolve = t_backsolve / ntrials,
    )
end

"""
    timing_madnlp(ips::InteriorPointSolver; ntrials=10)

Return the average time spent in the callbacks and in the linear solver,
for `ntrials` different trials.

Results are returned as a named-tuple.

"""
function timing_madnlp(ips; ntrials=10)
    return (
        time_linear_solver=timing_linear_solver(ips; ntrials=ntrials),
        time_callbacks=timing_callbacks(ips; ntrials=ntrials),
    )
end

