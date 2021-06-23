# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

using Pkg.Artifacts, BinaryProvider

const verbose = "--verbose" in ARGS
const prefix = Prefix(@__DIR__)
const so = BinaryProvider.platform_dlext()
const rpath = `-Wl,-rpath,`
const whole_archive= Sys.isapple() ? `-Wl,-all_load` : `-Wl,--whole-archive`
const no_whole_archive = Sys.isapple() ? `-Wl,-noall_load` : `-Wl,--no-whole-archive`
const libdir     = mkpath(joinpath(@__DIR__, "lib"))
const pardiso_library_path = haskey(ENV,"MADNLP_PARDISO_LIBRARY_PATH") ? ENV["MADNLP_PARDISO_LIBRARY_PATH"] : ""
const CC = haskey(ENV,"MADNLP_CC") ? ENV["MADNLP_CC"] : `gcc`
const libopenblas_dir = joinpath(artifact"OpenBLAS32","lib")
const with_openblas = `-L$libopenblas_dir $rpath$libopenblas_dir -lopenblas`

rm(libdir;recursive=true,force=true)
mkpath(libdir)
isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)

# Pardiso
if pardiso_library_path != ""
    if isvalid(`$CC --version`)
        ff = splitext(basename(pardiso_library_path)[4:end])[1]
        dd = dirname(pardiso_library_path)
        product = FileProduct(prefix,joinpath(libdir,"libpardiso.$so"),:libpardiso)
        wait(OutputCollector(`$CC -shared -olib/libpardiso.$so .pardiso_dummy.c $whole_archive -L$dd $rpath$dd -l$ff $no_whole_archive $with_openblas -lgfortran -fopenmp -lpthread -lm`,verbose=verbose))
        Sys.isapple() && satisfied(product) && wait(OutputCollector(`install_name_tool -change lib$name.$so @rpath/lib$name.$so lib/libpardiso.$so`,verbose=verbose))
    end
end

if satisfied(product)
    @info "Building Pardiso succeded."
    write_deps_file(joinpath(@__DIR__, "deps.jl"),[product], verbose=verbose)
else
    error("Building Pardiso failed.")
end

# write deps.jl
