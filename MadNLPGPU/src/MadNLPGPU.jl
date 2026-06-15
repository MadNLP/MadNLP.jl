module MadNLPGPU

# Backend-agnostic GPU support for the MadNLP interior-point solver: GPU methods
# of the IPM error/restoration kernels (get_varphi, get_inf_*, populate_RR_nn!,
# ...), the GPU bound counter, and the GPU scaling getters (get_sd/get_sc). These
# dispatch on AbstractGPUVectorOrSubVector / AbstractGPUVector (device-generic), so
# both cuMadNLP (CUDA) and rocMadNLP (ROCm) build on this shared base and add only
# their backend-specific MadNLPOptions constructor + linear-solver defaults.
# Migrated from MadNLPGPU/src/IPM (the device-agnostic part).

using Reexport
@reexport using MadNLP
import MadNLP:
    _get_varphi, get_varphi, get_inf_du, get_inf_compl, get_min_complementarity,
    get_varphi_d, get_alpha_max, get_alpha_z, get_obj_val_R, get_theta_R, get_inf_pr_R,
    get_inf_du_R, get_inf_compl_R, get_alpha_max_R, get_alpha_z_R, get_varphi_R, get_F,
    get_varphi_d_R, get_rel_search_norm, populate_RR_nn!, count_lu_bounds

import MadCoreKernelAbstractions: AbstractGPUVectorOrSubVector
import GPUArraysCore: AbstractGPUVector

include("kernels.jl")
include("utils.jl")
include("scaling.jl")

end # module
