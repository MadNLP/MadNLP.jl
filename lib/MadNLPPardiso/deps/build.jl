# remove deps.jl if it exists, in case build.jl fails
isfile("deps.jl") && rm("deps.jl")

using Libdl


println("Pardiso library")
println("===============")

const LIBPARDISONAMES =
if Sys.iswindows()
[
    "libpardiso.dll",
    "libpardiso600-WIN-X86-64.dll",
]
elseif Sys.isapple()
[
    "libpardiso.dylib",
    "libpardiso600-MACOS-X86-64.dylib",
]
elseif Sys.islinux()
[
    "libpardiso.so",
    "libpardiso600-GNU800-X86-64.so",
]
else
    error("unhandled OS")
end

println("Looking for libraries with name: ", join(LIBPARDISONAMES, ", "), ".")


PATH_PREFIXES = [@__DIR__; get(ENV, "JULIA_PARDISO", [])]

if !haskey(ENV, "JULIA_PARDISO")
    println("INFO: use the `JULIA_PARDISO` environment variable to set a path to " *
            "the folder where the Pardiso library is located")
end

function find_paradisolib()
    found_lib = false
    for prefix in PATH_PREFIXES
        println("Looking in \"$(abspath(prefix))\" for libraries")
        for libname in LIBPARDISONAMES
            local path
            try
                path = joinpath(prefix, libname)
                if isfile(path)
                    println("    found \"$(abspath(path))\", attempting to load it...")
                    Libdl.dlopen(path, Libdl.RTLD_GLOBAL)
                    println("    loaded successfully!")
                    global PARDISO_LIB_FOUND = true
                    return path, true
                end
            catch e
                println("    failed to load due to:")
                Base.showerror(stderr, e)
            end
        end
    end
    println("did not find libpardiso, assuming PARDISO 5/6 is not installed")
    return "", false
end

pardisopath, found_pardisolib = find_paradisolib()

#################################################

println("\nMKL Pardiso")
println("=============")
function find_mklparadiso()
    if haskey(ENV, "MKLROOT")
        println("found MKLROOT environment variable, enabling local MKL")
        return true
    end
    println("did not find MKLROOT environment variable, using MKL_jll")
    return false
end

found_mklpardiso = find_mklparadiso()

open("deps.jl", "w") do f
    print(f,
"""
const LOCAL_MKL_FOUND = $found_mklpardiso
const PARDISO_LIB_FOUND = $found_pardisolib
const PARDISO_PATH = raw"$pardisopath"
"""
)

end

