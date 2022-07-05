## Install
```julia
pkg> add MadNLPHSL
```

## Build

To build MadNLP with HSL linear solvers (Ma27, Ma57, Ma77, Ma86, Ma97), the source codes need to be obtained by the user from <http://www.hsl.rl.ac.uk/ipopt/> under Coin-HSL Full (Stable). The source codes are distribted as a tarball file `coinhsl-*.tar.gz`. The absolute path to the extracted source code or the complied library should be provided to the user. If the user has an already compiled HSL sovler library, one can simply provide a path to that shared library.In this case, the source code is not compiled and the provided shared library is directly used.
```julia
# at least one of the following should be given
julia> ENV["MADNLP_HSL_SOURCE_PATH"] = "/opt/coinhsl" 
julia> ENV["MADNLP_MA27_SOURCE_PATH"] = "/opt/coinhsl-archive-2021.05.05" 
julia> ENV["MADNLP_MA57_SOURCE_PATH"] = "/opt/ma57-3.11.0/" 
julia> ENV["MADNLP_MA77_SOURCE_PATH"] = "/opt/hsl_ma77-6.3.0" 
julia> ENV["MADNLP_MA86_SOURCE_PATH"] = "/opt/hsl_ma86-1.7.2" 
julia> ENV["MADNLP_MA97_SOURCE_PATH"] = "/opt/hsl_ma97-2.7.1" 

julia> ENV["MADNLP_HSL_LIBRARY_PATH"] = "/usr/lib/libcoinhsl.so"
julia> ENV["MADNLP_MA27_LIBRARY_PATH"] = "/usr/lib/libma27.so"
julia> ENV["MADNLP_MA57_LIBRARY_PATH"] = "/usr/lib/libma57.so"
julia> ENV["MADNLP_MA77_LIBRARY_PATH"] = "/usr/lib/libma77.so"
julia> ENV["MADNLP_MA86_LIBRARY_PATH"] = "/usr/lib/libma86.so"
julia> ENV["MADNLP_MA97_LIBRARY_PATH"] = "/usr/lib/libma97.so"
# optionally, one can specify
julia> ENV["MADNLP_HSL_BLAS"] = "mkl" # default is "openblas"
```
After obtaining the source code or the libarary, run
```julia
pkg> build MadNLPHSL
```

If HSL is built from the source code, the build process requires a Fortran compiler. If they are not installed, do:
```julia
shell> sudo apt install gfortran # Ubuntu, Debian
shell> brew cask install gfortran # MacOS
```
The compiler can be customized by:
```julia
julia> ENV["MADNLP_FC"] = "/usr/local/bin/gfortran" # default is "gfortran"
```

