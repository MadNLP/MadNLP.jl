To run benchmarks, run the following script.
```
# to compare current version and master using 10 cores and CUTEst and PowerModels.jl problems,
julia runbenchmarks.jl 10 power cutest current master
```

By default, logs are deactivated. To active them, please add `verbose` to the options:
```
julia runbenchmarks.jl 10 power cutest current master verbose
```
