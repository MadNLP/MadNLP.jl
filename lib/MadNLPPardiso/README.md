## Install
```julia
pkg> add MadNLPPardiso
```

## Build

To use Pardiso, the user needs to obtain the Paridso shared libraries from <https://www.pardiso-project.org/>, provide the absolute path to the shared library:
julia> ENV["MADNLP_PARDISO_LIBRARY_PATH"] = "/usr/lib/libpardiso600-GNU800-X86-64.so"
and place the license file in the home directory. 
```
After obtaining the libarary and the license file, run
```julia
pkg> build MadNLPPardiso
```

The build process requires a C compiler. If they are not installed, do:
```julia
shell> sudo apt install gcc # Ubuntu, Debian
shell> brew cask install gcc # MacOS
```
The compiler can be customized by:
```julia
julia> ENV["MADNLP_CC"] = "/usr/local/bin/gcc-9"    # default is "gcc"
```


