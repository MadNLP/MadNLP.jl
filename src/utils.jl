# MadNLP.jl.
# Created by Sungho Shin (sungho.shin@wisc.edu)

# Build info
function available_linear_solvers()
    solvers = ["umfpack"]
    @isdefined(libhsl) && append!(solvers,["Ma27","Ma57","Ma77","Ma86","Ma97"])
    @isdefined(libmumps) && push!(solvers,"Mumps")
    @isdefined(libpardiso) && push!(solvers,"Pardiso")
    @isdefined(libpardisomkl) && push!(solvers,"PardisoMKL")
    return solvers
end
default_linear_solver() = @isdefined(libhsl) ? Ma57 : @isdefined(libmumps) ? Mumps : Umfpack
default_subproblem_solver() = @isdefined(libhsl) ? Ma57 : Umfpack
default_dense_solver() = LapackMKL

# Options
abstract type AbstractOptions end
parse_option(::Type{Module},str::String) = eval(Symbol(str))
parse_option(::Type{Bool},str::String) = str == "yes" ? true : false
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
mutable struct MadNLPRecord <: AttributeRecord
    level::Attribute
    levelnum::Attribute
    msg::Attribute
    name::Attribute
end
MadNLPRecord(name::AbstractString, level::AbstractString, levelnum::Int, msg) = MadNLPRecord(
    Attribute(level),Attribute(levelnum),Attribute{AbstractString}(msg),Attribute(name))
mutable struct MadNLPHandler{F<:Formatter, O <: IO}  <: Handler{F}
    fmt::F
    io::O
    levelnum::Int
    color_dict::Dict{String,Symbol}
end
const default_color_dict = Dict{String,Symbol}(
    "trace" => :cyan,"debug" => :green,"info" => :normal,"notice" => :normal,"warn" => :magenta,"error" => :red)
const mono_color_dict = Dict{String,Symbol}(
    "trace" => :normal,"debug" => :normal,"info" => :normal,"notice" => :normal,"warn" => :normal,"error" => :normal)
function emit(handler::MadNLPHandler{F, O}, rec::Record) where {F<:Formatter, O<:IO}
    rec.levelnum < handler.levelnum && return
    str = format(handler.fmt, rec)
    clr = handler.color_dict[rec.level]
    clr==:normal ? println(handler.io, str) : printstyled(handler.io, str, "\n", color=clr)
    flush(handler.io)
end
const LOGGER = getlogger(@__MODULE__)
setpropagating!(LOGGER,false)
setrecord!(LOGGER,MadNLPRecord)

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

