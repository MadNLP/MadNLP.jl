# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

using Pkg.Artifacts, BinaryProvider, OpenBLAS32_jll

const verbose = "--verbose" in ARGS
const prefix = Prefix(@__DIR__)
const so = BinaryProvider.platform_dlext()
const rpath = `-Wl,-rpath,`
const whole_archive= Sys.isapple() ? `-Wl,-all_load` : `-Wl,--whole-archive`
const no_whole_archive = Sys.isapple() ? `-Wl,-noall_load` : `-Wl,--no-whole-archive`
const libdir     = mkpath(joinpath(@__DIR__, "lib"))
const hsl_shared_library = haskey(ENV,"MADNLP_HSL_LIBRARY_PATH") ? ENV["MADNLP_HSL_LIBRARY_PATH"] : ""
const pardiso_shared_library = haskey(ENV,"MADNLP_PARDISO_LIBRARY_PATH") ? ENV["MADNLP_PARDISO_LIBRARY_PATH"] : ""
const CC = haskey(ENV,"MADNLP_CC") ? ENV["MADNLP_CC"] : `gcc`
const openmp_flag = haskey(ENV,"MADNLP_ENABLE_OPENMP") ? ENV["MADNLP_ENABLE_OPENMP"] : `-fopenmp`
const libopenblas_dir = splitdir(OpenBLAS32_jll.libopenblas_path)[1]
const with_openblas = `-L$libopenblas_dir $rpath$libopenblas_dir -lopenblas`

rm.(filter(endswith(".so"), readdir(libdir,join=true)))
products   = Product[]
build_succeded(product::Product)=satisfied(product) ? "succeeded" : "failed"
isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)

# Pardiso
if pardiso_shared_library == ""
    if isvalid(`$CC --version`)
        const libpardiso_dir = joinpath(@__DIR__,"downloads")
        push!(products,FileProduct(prefix,joinpath(libdir,"libpardiso.$so"),:libpardiso))
        for name in readdir(libpardiso_dir)[occursin.("libpardiso",readdir(libpardiso_dir))]
            startswith(name,"lib") && endswith(name,so) ? name = splitext(name)[1][4:end] : continue
            with_pardiso=`-L$libpardiso_dir $rpath$libpardiso_dir -l$name`
            wait(OutputCollector(`$CC -shared -olib/libpardiso.$so .pardiso_dummy.c $whole_archive $with_pardiso $no_whole_archive $with_openblas -lgfortran $openmp_flag -lpthread -lm`,verbose=verbose))
            Sys.isapple() && satisfied(products[end]) &&
                wait(OutputCollector(`install_name_tool -change lib$name.$so @rpath/lib$name.$so lib/libpardiso.$so`,verbose=verbose))
            satisfied(products[end]) && break
        end
        @info "Building Pardiso $(build_succeded(products[end]))."
    end
else
    push!(products,FileProduct(pardiso_shared_library, :libpardiso))
    @info "Building Pardiso $(build_succeded(products[end]))."
end

# write deps.jl
write_deps_file(joinpath(@__DIR__, "deps.jl"), products[satisfied.(products)], verbose=verbose)
