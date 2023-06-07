## Install
```julia
pkg> add MadNLPHSL
```
Obtain a license and download HSL_jll.jl from https://licences.stfc.ac.uk/product/julia-hsl.

There are two versions available: LBT and OpenBLAS. LBT is the recommended option for Julia >= v1.9.

Install this download into your current environment using:

```julia
import Pkg
Pkg.develop(path = "/full/path/to/HSL_jll.jl")
```
