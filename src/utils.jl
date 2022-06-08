# MadNLP.jl.
# Created by Sungho Shin (sungho.shin@wisc.edu)
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

# Type definitions
SubVector{Tv}=SubArray{Tv, 1, Vector{Tv}, Tuple{Vector{Int}}, false}
StrideOneVector{Tv}=Union{
    Vector{Tv},
    SubArray{Tv,1,Vector{Tv},Tuple{UnitRange{Int}},true},
    SubArray{Tv, 1, Matrix{Tv}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}
}


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
