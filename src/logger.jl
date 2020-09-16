# MadNLP.jl.
# Created by Sungho Shin (sungho.shin@wisc.edu)

# Options
abstract type AbstractOptions end
parse_option(::Module,str::String) = eval(Symbol(str))
parse_option(::Bool,str::String) = str == "yes" ? true : false
function set_options!(opt::AbstractOptions,option_dict::Dict{Symbol,Any})
    for (key,val) in option_dict
        hasproperty(opt,key) || continue
        T = fieldtype(typeof(opt),key)
        val isa T ? setproperty!(opt,key,val) :
            setproperty!(opt,key,parse_option(getproperty(opt,key),val))
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
import Memento: Attribute, emit

const LOGGER = getlogger(@__MODULE__)
Memento.setpropagating!(LOGGER,false)
Memento.setrecord!(LOGGER,MadNLPRecord)

mutable struct MadNLPRecord <: AttributeRecord
    level::Attribute
    levelnum::Attribute
    msg::Attribute
    name::Attribute
end
function MadNLPRecord(name::AbstractString, level::AbstractString, levelnum::Int, msg)
    return MadNLPRecord(
        Attribute(level),Attribute(levelnum),
        Attribute{AbstractString}(msg),Attribute(name))
end
mutable struct MadNLPHandler{F<:Formatter, O <: IO}  <: Handler{F}
    fmt::F
    io::O
    levelnum::Int
    color_dict::Dict{String,Symbol}
end
const default_color_dict = Dict{String,Symbol}(
    "trace" => :gray,"debug" => :gray,"info" => :normal,
    "notice" => :normal,"warn" => :magenta,"error" => :red)
const mono_color_dict = Dict{String,Symbol}(
    "trace" => :normal,"debug" => :normal,"info" => :normal,
    "notice" => :normal,"warn" => :normal,"error" => :normal)
function emit(handler::MadNLPHandler{F, O}, rec::Record) where {F<:Formatter, O<:IO}
    rec.levelnum < handler.levelnum && return
    str = Memento.format(handler.fmt, rec)
    clr = handler.color_dict[rec.level]
    clr==:normal ? println(handler.io, str) : printstyled(handler.io, str,"\n", color=clr)
    flush(handler.io)
end
