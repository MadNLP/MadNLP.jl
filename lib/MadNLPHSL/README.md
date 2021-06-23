## Install
```julia
pkg> add MadNLPHSL
```

## Build

To build MadNLP with HSL linear solvers (Ma27, Ma57, Ma77, Ma86, Ma97), the source codes need to be obtained by the user from <http://www.hsl.rl.ac.uk/ipopt/> under Coin-HSL Full (Stable). The source codes are distribted as a tarball file `coinhsl-*.tar.gz`. The absolute path to the extracted source code or the complied library should be provided to the user. If the user has an already compiled HSL sovler library, one can simply provide a path to that shared library.In this case, the source code is not compiled and the provided shared library is directly used.
```julia
# either one of the following should be given
julia> ENV["MADNLP_HSL_SOURCE_PATH"] = "/opt/coinhsl" 
julia> ENV["MADNLP_HSL_LIBRARY_PATH"] = "/usr/lib/libcoinhsl.so"
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

