module MadNLPBenchmark

using Logging, CUDA, ArgParse, Distributed, JLD2

using ExaModelsPower

include("common.jl")
include("opf.jl")

export benchmark_opf, benchmark_cops, benchmark_cutest

end
