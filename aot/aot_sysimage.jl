# Create an AOT-compiled system image with MadNLP
#
# This script uses PackageCompiler.jl to create a system image that
# includes precompiled MadNLP code for faster startup.
#
# Usage:
#   julia --project=. aot/aot_sysimage.jl
#
# After creation:
#   julia --sysimage=MadNLP_sysimage.so -e 'using MadNLP; println(MadNLP.introduce())'
#
# For a fully static executable (Julia 1.12+), see aot_example.jl.

using PackageCompiler

# Precompile statements are gathered by running the example
create_sysimage(
    [:MadNLP, :NLPModels];
    sysimage_path="MadNLP_sysimage." * Libdl.dlext,
    precompile_execution_file=joinpath(@__DIR__, "aot_example.jl"),
)

println("System image created successfully!")
println("Run with: julia --sysimage=MadNLP_sysimage.", Libdl.dlext)
