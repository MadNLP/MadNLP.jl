using Pkg.Artifacts, BinaryProvider

if haskey(ENV,"MADNLP_HSL_BLAS") && ENV["MADNLP_HSL_BLAS"]=="mkl"
     blasvendor = :mkl
else
     blasvendor = :openblas
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
const FC = haskey(ENV,"MADNLP_FC") ? ENV["MADNLP_FC"] : `gfortran`
const libmetis_dir = joinpath(artifact"METIS", "lib")
const with_metis = `-L$libmetis_dir $rpath$libmetis_dir -lmetis`
if blasvendor == :mkl 
    const libblas_dir = joinpath(artifact"MKL","lib")
    const libopenmp_dir = joinpath(artifact"IntelOpenMP","lib")
    const with_blas = `-L$libblas_dir $rpath$libblas_dir -lmkl_rt -L$libopenmp_dir $rpath$libopenmp_dir -liomp5`
else
    const libblas_dir = joinpath(artifact"OpenBLAS32","lib")
    const with_blas = `-L$libblas_dir $rpath$libblas_dir -lopenblas`
end
    
const targets = [
    "ddeps.f",  "fakemetis.f",  "sdeps.f",
    "deps.f", "deps90.f90", "dump.f90", "hsl_mc68i_ciface.f90",
    "ma27d.f", "ma27s.f", "ma57d.f", "ma57s.f",
    "hsl_ma77d.f90", "hsl_ma77d_ciface.f90",
    "hsl_ma77s.f90", "hsl_ma77s_ciface.f90",
    "hsl_ma86d.f90", "hsl_ma86d_ciface.f90",
    "hsl_ma86s.f90", "hsl_ma86s_ciface.f90",
    "hsl_ma97d.f90", "hsl_ma97d_ciface.f90", 
    "hsl_ma97s.f90", "hsl_ma97s_ciface.f90", 
]

rm(libdir;recursive=true,force=true)
mkpath(libdir)
isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)


# HSL
if hsl_source_path != ""
    if isvalid(`$FC --version`)
        OC = OutputCollector[]
        cd(hsl_source_path)

        names = []
        for (root, dirs, files) in walkdir(hsl_source_path)
            for file in files;
                if file in targets;
                    name = splitext(relpath(joinpath(root,file),hsl_source_path))
                    push!(names, name)
                    @info "$(name[1])$(name[2]) source code detected."
                end
            end
        end
        
        succeed = wait.(
            [OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o $name.o $name$ext`,verbose=verbose)
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

