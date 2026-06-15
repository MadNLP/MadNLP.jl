module rocMadNLP

# ROCm/AMDGPU backend layer for MadNLP's interior-point solver. The device-agnostic
# GPU IPM kernels/scaling live in the shared MadNLPGPU base (re-exported here); this
# package adds the AMD-specific GPU-default MadNLPOptions constructor. AMD has no
# GPU sparse Cholesky (no cuDSS analog), so the GPU linear solver is always the
# dense LapackROCmSolver. Migrated from MadNLPGPU/ext/MadNLPGPUAMDGPUExt.

using Reexport
@reexport using MadNLPGPU
import MadNLP  # options.jl extends MadNLP.MadNLPOptions

import AMDGPU: ROCVector, ROCBackend
import MadCoreAMDGPU: LapackROCmSolver

include("options.jl")

# Re-export the ROCm KernelAbstractions backend so `using rocMadNLP` gives users
# ROCBackend() for constructing GPU KKT systems.
export ROCBackend

end # module
