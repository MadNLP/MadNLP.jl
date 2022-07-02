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

const supported_library = [
    (:libhsl, "MADNLP_COINHSL_LIBRARY_PATH", "MADNLP_COINHSL_PATH")
    (:libma27, "MADNLP_MA27_LIBRARY_PATH", "MADNLP_MA27_PATH")
    (:libma57, "MADNLP_MA57_LIBRARY_PATH", "MADNLP_MA57_PATH")
    (:libma77, "MADNLP_MA77_LIBRARY_PATH", "MADNLP_MA77_PATH")
    (:libma86, "MADNLP_MA86_LIBRARY_PATH", "MADNLP_MA86_PATH")
    (:libma97, "MADNLP_MA97_LIBRARY_PATH", "MADNLP_MA97_PATH")
]

const targets_dict = Dict(
    :libhsl=> [
        [
            "deps.f",
            "deps90.f90",
        ],
        [
            "ma27d.f",
            "ma57d.f",
            "hsl_ma77d.f90",
            "hsl_ma86d.f90",
            "hsl_ma97d.f90",
        ],
        [
            "hsl_mc68i_ciface.f90",
            "hsl_ma77d_ciface.f90",
            "hsl_ma86d_ciface.f90",
            "hsl_ma97d_ciface.f90",
        ]
    ],
    :libma27 => [
        [
            "ma27d.f",
            "ma27s.f",
        ],
    ],
    :libma57 => [
        [
            "sdeps.f", "ddeps.f", 
        ],
        [
            "ma57d.f", "ma57s.f", 
        ],
    ],
    :libma77 => [
        [
            "sdeps.f", "ddeps.f", 
        ],
        [
            "ma57d.f", "ma57s.f", 
        ],
    ],
    :libma86 => [
        [
            "sdeps.f", "ddeps.f", 
        ],
        [
            "ma57d.f", "ma57s.f", 
        ],
    ],
    :libma97 => [
        [
            "sdeps.f", "ddeps.f", 
        ],
        [
            "ma97d.f", "ma97s.f", 
        ],
    ]
)

rm(libdir;recursive=true,force=true)
mkpath(libdir)
isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)


# HSL
products = Product[]
for (lib, envsrc, envlib) in supported_library
    if haskey(ENV,envlib)
        push!(products, FileProduct(ENV[envlib], lib))
    elseif haskey(ENV,envsrc) && isvalid(`$FC --version`)
        source_path = ENV[envsrc]
        targets = targets_dict[lib]
        
        OC = OutputCollector[]
        cd(source_path)

        names_succeeded = []
        for target in targets
            names = []
            for (root, dirs, files) in walkdir(source_path)
                for file in files
                    if file in target
                        filter!(x->x != file,files)
                        name = splitext(relpath(joinpath(root,file),source_path))
                        push!(names, name)
                        @info "$(name[1])$(name[2]) source code detected."
                    end
                end
            end
            succeeded = wait.(
                [OutputCollector(`$FC -fopenmp -fPIC -c -O3 -o $name.o $name$ext`,verbose=verbose)
                 for (name,ext) in names])
            append!(names_succeeded, names[succeeded])
        end

        cmd = `$FC -o$(libdir)/$lib.$so -shared -fPIC -O3 -fopenmp`
        append!(cmd.exec, ["$name.o" for (name,ext) in names_succeeded])
        append!(cmd.exec, with_metis.exec)
        append!(cmd.exec, with_blas.exec)
        
        run(cmd)
        cd("$(@__DIR__)")
        push!(products, FileProduct(prefix,joinpath(libdir,"$lib.$so"), lib))
    end
end

# write deps.jl
products_succeeded = Product[]
for product in products
    if satisfied(product)
        @info "Building HSL succeeded."
        push!(products_succeded, product)
    else
        @error "Building HSL failed."
    end
end

write_deps_file(joinpath(@__DIR__, "deps.jl"),products_succeded, verbose=verbose)
