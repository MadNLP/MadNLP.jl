# MadNLPHSL.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

using Pkg.Artifacts, BinaryProvider

const verbose = "--verbose" in ARGS
const prefix = Prefix(@__DIR__)
const so = BinaryProvider.platform_dlext()
const rpath = `-Wl,-rpath,`
const whole_archive= Sys.isapple() ? `-Wl,-all_load` : `-Wl,--whole-archive`
const no_whole_archive = Sys.isapple() ? `-Wl,-noall_load` : `-Wl,--no-whole-archive`
const libdir     = mkpath(joinpath(@__DIR__, "lib"))
const hsl_library_path = haskey(ENV,"MADNLP_HSL_LIBRARY_PATH") ? ENV["MADNLP_HSL_LIBRARY_PATH"] : ""
const hsl_source_path = haskey(ENV,"MADNLP_HSL_SOURCE_PATH") ? ENV["MADNLP_HSL_SOURCE_PATH"] : ""
const hsl_version = haskey(ENV,"MADNLP_HSL_VERSION_NUMBER") ? ENV["MADNLP_HSL_VERSION_NUMBER"] : "2015.06.23"
const FC = haskey(ENV,"MADNLP_FC") ? ENV["MADNLP_FC"] : `gfortran`
const libmetis_dir = joinpath(artifact"METIS", "lib")
const with_metis = `-L$libmetis_dir $rpath$libmetis_dir -lmetis`
const libopenblas_dir = joinpath(artifact"OpenBLAS32","lib")
const with_openblas = `-L$libopenblas_dir $rpath$libopenblas_dir -lopenblas`
const installer = Sys.isapple() ? "brew install" : Sys.iswindows() ? "pacman -S" : "sudo apt install"

rm(libdir;recursive=true,force=true)
mkpath(libdir)
isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)

# HSL
if hsl_library_path == ""
    if isvalid(`$FC --version`)
        OC = OutputCollector[]
        cd(hsl_source_path)
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o common/deps.o common/deps.f`,verbose=verbose))
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o common/deps90.o common/deps90.f90`,verbose=verbose))
        wait.(OC); empty!(OC)
        push!(OC,OutputCollector(`$FC -fPIC -c -O3 -o mc19/mc19d.o mc19/mc19d.f`,verbose=verbose))
        push!(OC,OutputCollector(`$FC -fPIC -c -O3 -o ma27/ma27d.o ma27/ma27d.f`,verbose=verbose))
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o ma57/ma57d.o ma57/ma57d.f`,verbose=verbose))    
        wait.(OC); empty!(OC)
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma77/hsl_ma77d.o hsl_ma77/hsl_ma77d.f90`,verbose=verbose))
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma86/hsl_ma86d.o hsl_ma86/hsl_ma86d.f90`,verbose=verbose))
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma97/hsl_ma97d.o hsl_ma97/hsl_ma97d.f90`,verbose=verbose))
        wait.(OC); empty!(OC)
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma77/C/hsl_ma77d_ciface.o hsl_ma77/C/hsl_ma77d_ciface.f90`,verbose=verbose))
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma86/C/hsl_ma86d_ciface.o hsl_ma86/C/hsl_ma86d_ciface.f90`,verbose=verbose))
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_mc68/C/hsl_mc68i_ciface.o hsl_mc68/C/hsl_mc68i_ciface.f90`,verbose=verbose))
        push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma97/C/hsl_ma97d_ciface.o hsl_ma97/C/hsl_ma97d_ciface.f90`,verbose=verbose))
        wait.(OC); empty!(OC)
        wait(OutputCollector(`$FC -o$(libdir)/libhsl.$so -shared -fPIC -O3 common/deps90.o common/deps.o mc19/mc19d.o ma27/ma27d.o ma57/ma57d.o hsl_ma77/hsl_ma77d.o hsl_ma77/C/hsl_ma77d_ciface.o hsl_ma86/hsl_ma86d.o hsl_ma86/C/hsl_ma86d_ciface.o hsl_mc68/C/hsl_mc68i_ciface.o hsl_ma97/hsl_ma97d.o hsl_ma97/C/hsl_ma97d_ciface.o -fopenmp $with_metis $with_openblas`,verbose=verbose))
        cd("$(@__DIR__)")
        product = FileProduct(prefix,joinpath(libdir,"libhsl.$so"), :libhsl)
    end
else
    product = FileProduct(hsl_library_path, :libhsl)
end

# write deps.jl
if satisfied(product)
    @info "Building HSL (Ma27, Ma57, Ma77, Ma86, Ma97) succeeded."
    write_deps_file(joinpath(@__DIR__, "deps.jl"),[product], verbose=verbose)
else
    @info "Building HSL (Ma27, Ma57, Ma77, Ma86, Ma97) failed."
end
