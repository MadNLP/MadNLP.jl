# Release notes

## v0.5.2 (February 27th, 2023)

### Bug fixes

- Fix URL in README (#236)
- Fix invalid return code in MOI.TerminationStatus (#241)
- Fix potential infinite loop in feasibility restoration (#244)
- Fix initial multipliers when initial KKT has no solution (#243)

### Performance and maintenance

- Replace broadcast operators by explicit for loops in IPM kernels to reduce time-to-first-solve (#231)
- Remove allocations in nonlinear callbacks (#230)


## v0.5.1 (October 20th, 2022)

### New features

- Update MOI wrapper to match new Ipopt's MOI wrapper (#224, #233)
- Add new constructor for custom KKT type (#232)

### Bug fixes

- Fix detection of number of upper and lower bounds for optimization variables (#211)
- Fix `solve!` function when dual is provided as input (#215)
- Update initialization of meta field (#216)
- Fix type stability in MadNLP (#220, #227, #228)

### Performance and maintenance

- Doc: Fix typos in quickstart.md (#210)
- Test properly `solve!` function (#219)
- Improve error messages when invalid number is detected (#226)


## v0.5.0 (August 30th, 2022)

### Breaking changes

- Change names to follow JSO conventions
  - `AbstractInteriorPointSolver` -> `AbstractMadNLPSolver`
  - `InteriorPointSolver` -> `MadNLPSolver`
  - `IPMOptions` -> `MadNLPOptions`
  - `Counters` -> `MadNLPCounters`
  - `Logger` -> `MadNLPLogger`
- Linear solvers are now defined as a struct, not a module
  - `MadNLPLapackCPU.Solver` -> `LapackCPUSolver`
  - `MadNLPUmfpack.Solver` -> `UmfpackSolver`
  - `MadNLPLapackGPU.Solver` -> `LapackGPUSolver`
  - `MadNLPMa27.Solver` -> `Ma27Solver`
  - `MadNLPMa57.Solver` -> `Ma57Solver`
  - `MadNLPMa77.Solver` -> `Ma77Solver`
  - `MadNLPMa86.Solver` -> `Ma86Solver`
  - `MadNLPMa97.Solver` -> `Ma97Solver`
  - `MadNLPKrylov.Solver` -> `KrylovIterator`
  - `MadNLPMumps.Solver` -> `MumpsSolver`
  - `MadNLPPardiso.Solver` -> `PardisoSolver`
  - `MadNLPPardisoMKL.Solver` -> `PardisoMKLSolver`
- Refactor the way we pass options to MadNLP
  - *Before:* `ips = MadNLP.InteriorPointSolver(nlp; option_dict=sparse_options)`
  - *Now:* `ips = MadNLP.MadNLPSolver(nlp; sparse_options...)`

### New features

- Add support for `Float32`
- Add support for NLSModels
- Add a function `timing_madnlp` to decompose the time spent in the callbacks and in the linear solver at each iteration
- Add a `CuMadNLPSolver` constructor to instantiate MadNLP on CUDA GPU

### Bug fixes

- Stop overwriting the number of threads used in BLAS
- Use symmetric `mul!` when using `DENSE_CONDENSED_KKT_SYSTEM` on the GPU
- Fix unscaling of the constraints during post-processing

### Performance and maintenance

- Add a toy nonlinear problem `HS15Model` in MadNLPTests
- Improve the build of MadNLPHSL
- Remove `StrideOneVector` alias in MadNLP
- Update the documentation
- Improve code coverage

