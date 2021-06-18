# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

using Pkg.Artifacts, BinaryProvider, OpenBLAS32_jll, MKL_jll

if haskey(ENV,"MADNLP_BLAS")
    blasvendor = ENV["MADNLP_BLAS"]=="openblas" ? :openblas : :mkl
else
    blasvendor = MKL_jll.best_platform == nothing ? :openblas : :mkl
end

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
const FC = haskey(ENV,"MADNLP_FC") ? ENV["MADNLP_FC"] : `gfortran`
const openmp_flag = haskey(ENV,"MADNLP_ENABLE_OPENMP") ? ENV["MADNLP_ENABLE_OPENMP"] : `-fopenmp`
const optimization_flag = haskey(ENV,"MADNLP_OPTIMIZATION_FLAG") ? ENV["MADNLP_OPTIMIZATION_FLAG"] : `-O3`
const installer = Sys.isapple() ? "brew install" : Sys.iswindows() ? "pacman -S" : "sudo apt install"
const libopenblas_dir = splitdir(OpenBLAS32_jll.libopenblas_path)[1]
const with_openblas = `-L$libopenblas_dir $rpath$libopenblas_dir -lopenblas`
const libmkl_dir = joinpath(MKL_jll.artifact_dir,MKL_jll.libmkl_rt_splitpath[1:end-1]...)
const with_mkl = `-L$libmkl_dir $rpath$libmkl_dir -lmkl_rt`

rm.(filter(endswith(".so"), readdir(libdir,join=true)))
products   = Product[]
build_succeded(product::Product)=satisfied(product) ? "succeeded" : "failed"
isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)


@info "Building MadNLP with $(blasvendor == :mkl ? "MKL" : "OpenBLAS")"

# Pardiso
if pardiso_shared_library == ""
    if isvalid(`$CC --version`)
        const libpardiso_dir = joinpath(@__DIR__,"download")
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
