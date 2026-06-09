# STASHED for phase F / MadNLP/lib/CuMadNLP — IPM-specific GPU methods split out
# of MadNLPGPU/src/utils.jl (they extend MadNLP's interior-point get_sd/get_sc).
# In CuMadNLP these stay `MadNLP.get_sd`/`MadNLP.get_sc` (or import + extend),
# and need AbstractGPUVector (GPUArraysCore) in scope.
if VERSION > v"1.11"
    _my1norm(x) = mapreduce(abs, +, x)
    function MadNLP.get_sd(l::AbstractGPUVector{T}, zl_r, zu_r, s_max) where {T}
        return max(
            s_max,
            (_my1norm(l) + _my1norm(zl_r) + _my1norm(zu_r)) / max(1, (length(l) + length(zl_r) + length(zu_r))),
        ) / s_max
    end
    function MadNLP.get_sc(zl_r::SubArray{T, 1, VT}, zu_r, s_max) where {T, VT <: AbstractGPUVector{T}}
        return max(
            s_max,
            (_my1norm(zl_r) + _my1norm(zu_r)) / max(1, length(zl_r) + length(zu_r)),
        ) / s_max
    end
end
