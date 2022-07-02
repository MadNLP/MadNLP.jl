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
    (:libhsl, "MADNLP_HSL_LIBRARY_PATH", "MADNLP_HSL_SOURCE_PATH")
    (:libma27, "MADNLP_MA27_LIBRARY_PATH", "MADNLP_MA27_SOURCE_PATH")
    (:libma57, "MADNLP_MA57_LIBRARY_PATH", "MADNLP_MA57_SOURCE_PATH")
    (:libma77, "MADNLP_MA77_LIBRARY_PATH", "MADNLP_MA77_SOURCE_PATH")
    (:libma86, "MADNLP_MA86_LIBRARY_PATH", "MADNLP_MA86_SOURCE_PATH")
    (:libma97, "MADNLP_MA97_LIBRARY_PATH", "MADNLP_MA97_SOURCE_PATH")
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
            "deps.f",
            "ma27d.f",
            "ma27s.f",
        ],
    ],
    :libma57 => [
        [
            "fakemetis.f",
            "sdeps.f", "ddeps.f",
            "ma57d.f", "ma57s.f", 
        ],
    ],
    :libma77 => [
        [
            "common.f", "common90.f90",
            "ddeps90.f90", "sdeps90.f90", 
        ],
        [
            "hsl_ma77d.f90", "hsl_ma77s.f90",
        ],
        [
            "hsl_ma77d_ciface.f90", "hsl_ma77s_ciface.f90",
        ],
    ],
    :libma86 => [
        [
            "common.f", "common90.f90",
            "sdeps90.f90", "fakemetis.f",
        ],
        [
            "hsl_ma86d.f90", "hsl_ma86s.f90",
        ],
        [
            "hsl_ma86d_ciface.f90", "hsl_ma86s_ciface.f90",
            "hsl_mc68i_ciface.f90",
        ],
    ],
    :libma97 => [
        [
            "common.f", "common90.f90",
            "sdeps90.f90", "ddeps90.f90", "fakemetis.f",
        ],
        [
            "hsl_ma97d.f90", "hsl_ma97s.f90",
        ],
        [
            "hsl_ma97d_ciface.f90", "hsl_ma97s_ciface.f90",
        ],
    ]
)

rm(libdir;recursive=true,force=true)
mkpath(libdir)
isvalid(cmd::Cmd)=(try run(cmd) catch e return false end; return true)


# HSL
attempted = Tuple{Symbol,Product}[]

for (lib, envlib, envsrc) in supported_library
    if haskey(ENV,envlib)
        push!(attempted, (lib,FileProduct(ENV[envlib], lib)))
    elseif haskey(ENV,envsrc) && isvalid(`$FC --version`)
        @info "Compiling $lib"
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
        push!(attempted, (lib,FileProduct(prefix,joinpath(libdir,"$lib.$so"), lib)))
    end
end

# write deps.jl
succeeded = Product[]
for (lib, product) in attempted
    if satisfied(product)
        @info "Building $lib succeeded."
        push!(succeeded, product)
    else
        @error "Building $lib failed."
    end
end

write_deps_file(joinpath(@__DIR__, "deps.jl"), succeeded, verbose=verbose)
