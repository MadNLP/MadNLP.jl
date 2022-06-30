# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

include("MOI_interface.jl")

# Thin wrapper for NLPModels.jl
"""
TODO
"""
function madnlp(model::AbstractNLPModel;buffered=true, kwargs...)
    ips = InteriorPointSolver(model;kwargs...)
    initialize!(ips.kkt)
    return optimize!(ips)
end

