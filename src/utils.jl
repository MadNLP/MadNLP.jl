# MadNLP.jl.
# Created by Sungho Shin (sungho.shin@wisc.edu)

# Build info
default_linear_solver() = @isdefined(libhsl) ? Ma57 : @isdefined(libmumps) ? Mumps : Umfpack
default_dense_solver() = LapackMKL

# Options
abstract type AbstractOptions end
parse_option(::Type{Module},str::String) = eval(Symbol(str))
function set_options!(opt::AbstractOptions,option_dict::Dict{Symbol,Any})
    for (key,val) in option_dict
        hasproperty(opt,key) || continue
        T = fieldtype(typeof(opt),key)
        val isa T ? setproperty!(opt,key,val) :
            setproperty!(opt,key,parse_option(T,val))
        pop!(option_dict,key)
    end
end
function set_options!(opt::AbstractOptions,option_dict::Dict{Symbol,Any},kwargs)
    !isempty(kwargs) && (for (key,val) in kwargs; option_dict[key]=val; end)
    set_options!(opt,option_dict)
end

# Dummy module
module DummyModule end

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
const blas_num_threads = Ref{Int}()
function set_blas_num_threads(n::Integer;permanent::Bool=false)
    permanent && (blas_num_threads[]=n)
    BLAS.set_num_threads(n) # might be mkl64 or openblas64
    ccall((:mkl_set_dynamic, libmkl32),
          Cvoid,
          (Ptr{Int32},),
          Ref{Int32}(0))
    ccall((:mkl_set_num_threads, libmkl32),
          Cvoid,
          (Ptr{Int32},),
          Ref{Int32}(n))
    ccall((:openblas_set_num_threads, libopenblas32),
          Cvoid,
          (Ptr{Int32},),
          Ref{Int32}(n))
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
SubVector{T}=SubArray{T,1}
StrideOneVector{T}=Union{
    Vector{T},SubArray{T,1,Array{T,1},Tuple{UnitRange{U}},true} where {U<:Integer},
    SubArray{Float64,1,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{U}},U},true} where {U<:Integer}}

