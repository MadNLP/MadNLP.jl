module CuMadNLP

# IPM-specific GPU support for MadNLP on CUDA: GPU methods of the interior-point
# error/restoration kernels (get_varphi, get_inf_*, populate_RR_nn!, ...), the
# GPU bound counter, the GPU scaling getters (get_sd/get_sc), and the GPU-default
# MadNLPOptions constructor. Migrated from MadNLPGPU/src/IPM. Builds on MadNLP +
# MadCoreCUDA.
#
# NOTE: the generic-GPU IPM kernels here dispatch on AbstractGPUVectorOrSubVector
# (backend-agnostic), so a future RocMadNLP would want them too. For now they
# live here (CUDA is the tested backend); factor a shared base out when RocMadNLP
# is built.

import MadNLP
import MadNLP:
    _get_varphi, get_varphi, get_inf_du, get_inf_compl, get_min_complementarity,
    get_varphi_d, get_alpha_max, get_alpha_z, get_obj_val_R, get_theta_R, get_inf_pr_R,
    get_inf_du_R, get_inf_compl_R, get_alpha_max_R, get_alpha_z_R, get_varphi_R, get_F,
    get_varphi_d_R, get_rel_search_norm, populate_RR_nn!, count_lu_bounds

import MadCoreKernelAbstractions: AbstractGPUVectorOrSubVector
import GPUArraysCore: AbstractGPUVector
import CUDACore: CuVector
import MadCoreCUDA: LapackCUDASolver, CUDSSSolver

include("kernels.jl")
include("utils.jl")
include("scaling.jl")
include("options.jl")

end # module
