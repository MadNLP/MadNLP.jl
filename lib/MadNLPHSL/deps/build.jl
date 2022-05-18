# MadNLPHSL.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

using Pkg.Artifacts, BinaryProvider

 if haskey(ENV,"MADNLP_HSL_BLAS")
    blasvendor = (ENV["MADNLP_HSL_BLAS"]=="openblas") ? :openblas : :mkl
else
    blasvendor =  artifact_hash("MKL",joinpath(@__DIR__, "..", "Artifacts.toml")) != nothing ? :mkl : :openblas
 end

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
const libblas_dir = joinpath(blasvendor == :mkl ? artifact"MKL" : artifact"OpenBLAS32","lib")
const with_blas = blasvendor == :mkl ? `-L$libblas_dir $rpath$libblas_dir -lmkl_rt` :
    `-L$libblas_dir $rpath$libblas_dir -lopenblas`
const installer = Sys.isapple() ? "brew install" : Sys.iswindows() ? "pacman -S" : "sudo apt install"

rm(libdir;recursive=true,force=true)
mkpath(libdir)
isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)

# HSL
if hsl_source_path != ""
    if isvalid(`$FC --version`)
        OC = OutputCollector[]
        cd(hsl_source_path)

        if isdir("ma57") # coinhsl-full
            names = [
                ("common/deps","f"),
                ("common/deps90","f90"),
                ("ma27/ma27d","f"), 
                ("ma57/ma57d","f"), 
                ("hsl_ma77/hsl_ma77d","f90"),
                ("hsl_ma86/hsl_ma86d","f90"), 
                ("hsl_ma97/hsl_ma97d","f90"),
                ("hsl_mc68/C/hsl_mc68i_ciface","f90"), 
                ("hsl_ma77/C/hsl_ma77d_ciface","f90"), 
                ("hsl_ma86/C/hsl_ma86d_ciface","f90"), 
                ("hsl_ma97/C/hsl_ma97d_ciface","f90"), 
            ]
        else # coinhsl-archive
            names = [
                ("common/deps","f"),
                ("ma27/ma27d","f"), ("ma27/ma27s","f")
            ]
        end
        
        succeed = wait.(
            [OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o $name.o $name.$ext`,verbose=verbose)
             for (name,ext) in names])
        names_succeed = names[succeed]

        cmd = `$FC -o$(libdir)/libhsl.$so -shared -fPIC -O3 -fopenmp`
        append!(cmd.exec, ["$name.o" for (name,ext) in names_succeed])
        append!(cmd.exec, with_metis.exec)
        append!(cmd.exec, with_blas.exec)
        
        run(cmd)
        cd("$(@__DIR__)")
        product = FileProduct(prefix,joinpath(libdir,"libhsl.$so"), :libhsl)
    end
else
    product = FileProduct(hsl_library_path, :libhsl)
end
    

# write deps.jl
if satisfied(product)
    @info "Building HSL succeeded."
    write_deps_file(joinpath(@__DIR__, "deps.jl"),Product[product], verbose=verbose)
else
    @error "Building HSL failed."
    write_deps_file(joinpath(@__DIR__, "deps.jl"),Product[], verbose=verbose)
end
# # MadNLPHSL.jl
# # Created by Sungho Shin (sungho.shin@wisc.edu)

# using Pkg.Artifacts, BinaryProvider

# if haskey(ENV,"MADNLP_HSL_BLAS")
#     blasvendor = (ENV["MADNLP_HSL_BLAS"]=="openblas") ? :openblas : :mkl
# else
#     blasvendor =  artifact_hash("MKL",joinpath(@__DIR__, "..", "Artifacts.toml")) != nothing ? :mkl : :openblas
# end

# const verbose = "--verbose" in ARGS
# const prefix = Prefix(@__DIR__)
# const so = BinaryProvider.platform_dlext()
# const rpath = `-Wl,-rpath,`
# const whole_archive= Sys.isapple() ? `-Wl,-all_load` : `-Wl,--whole-archive`
# const no_whole_archive = Sys.isapple() ? `-Wl,-noall_load` : `-Wl,--no-whole-archive`
# const libdir     = mkpath(joinpath(@__DIR__, "lib"))
# const hsl_library_path = haskey(ENV,"MADNLP_HSL_LIBRARY_PATH") ? ENV["MADNLP_HSL_LIBRARY_PATH"] : ""
# const hsl_source_path = haskey(ENV,"MADNLP_HSL_SOURCE_PATH") ? ENV["MADNLP_HSL_SOURCE_PATH"] : ""
# const hsl_version = haskey(ENV,"MADNLP_HSL_VERSION_NUMBER") ? ENV["MADNLP_HSL_VERSION_NUMBER"] : "2015.06.23"
# const FC = haskey(ENV,"MADNLP_FC") ? ENV["MADNLP_FC"] : `gfortran`
# const libmetis_dir = joinpath(artifact"METIS", "lib")
# const with_metis = `-L$libmetis_dir $rpath$libmetis_dir -lmetis`
# const libblas_dir = joinpath(blasvendor == :mkl ? artifact"MKL" : artifact"OpenBLAS32","lib")
# const with_blas = blasvendor == :mkl ? `-L$libblas_dir $rpath$libblas_dir -lmkl_rt` :
#     `-L$libblas_dir $rpath$libblas_dir -lopenblas`
# const installer = Sys.isapple() ? "brew install" : Sys.iswindows() ? "pacman -S" : "sudo apt install"

# rm(libdir;recursive=true,force=true)
# mkpath(libdir)
# isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)

# # HSL
# if hsl_source_path != ""
#     if isvalid(`$FC --version`)
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o common/deps.o common/deps.f`,verbose=verbose))
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o common/deps90.o common/deps90.f90`,verbose=verbose))
#         # wait.(OC); empty!(OC)
#         # import BinaryProvider: wait
#         # struct NullOutputCollector end
#         # wait(::NullOutputCollector) = false
        
#         cd(hsl_source_path)
        

        
#         # wait(OutputCollector(`$FC -o$(libdir)/libhsl.$so -shared -fPIC -O3 common/deps.o ma27/ma27d.o ma27/ma27s.o -fopenmp $with_metis $with_blas`,verbose=verbose))

#         # push!(OC,OutputCollector(`$FC -fPIC -c -O3 -o ma27/ma27d.o ma27/ma27d.f`,verbose=verbose))
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o ma57/ma57d.o ma57/ma57d.f`,verbose=verbose))    
#         # wait.(OC); empty!(OC)
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma77/hsl_ma77d.o hsl_ma77/hsl_ma77d.f90`,verbose=verbose))
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma86/hsl_ma86d.o hsl_ma86/hsl_ma86d.f90`,verbose=verbose))
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma97/hsl_ma97d.o hsl_ma97/hsl_ma97d.f90`,verbose=verbose))
#         # wait.(OC); empty!(OC)
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma77/C/hsl_ma77d_ciface.o hsl_ma77/C/hsl_ma77d_ciface.f90`,verbose=verbose))
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma86/C/hsl_ma86d_ciface.o hsl_ma86/C/hsl_ma86d_ciface.f90`,verbose=verbose))
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_mc68/C/hsl_mc68i_ciface.o hsl_mc68/C/hsl_mc68i_ciface.f90`,verbose=verbose))
#         # push!(OC,OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o hsl_ma97/C/hsl_ma97d_ciface.o hsl_ma97/C/hsl_ma97d_ciface.f90`,verbose=verbose))
#         # wait.(OC); empty!(OC)

#         # cd("$(@__DIR__)")
#         product = FileProduct(prefix,joinpath(libdir,"libhsl.$so"), :libhsl)
#     end
# else
#     product = FileProduct(hsl_library_path, :libhsl)
# end

    

# # write deps.jl
# if satisfied(product)
#     @info "Building HSL (Ma27, Ma57, Ma77, Ma86, Ma97) succeeded."
#     write_deps_file(joinpath(@__DIR__, "deps.jl"),Product[product], verbose=verbose)
# else
#     @error "Building HSL (Ma27, Ma57, Ma77, Ma86, Ma97) failed."
#     write_deps_file(joinpath(@__DIR__, "deps.jl"),Product[], verbose=verbose)
# end
