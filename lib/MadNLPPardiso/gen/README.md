# Wrapping headers

This directory contains a script that can be used to automatically generate wrappers from C headers provided by [Panua-Pardiso](https://panua.ch/pardiso/).
This is done using Clang.jl.

# Setup

You need to create an `include` folder that contains the header files `pardiso.h`.

# Usage

Either run `julia wrapper.jl` directly, or include it and call the `main()` function.
Be sure to activate the project environment in this folder (`julia --project`), which will install `Clang.jl` and `JuliaFormatter.jl`.
