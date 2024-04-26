#=
    MadNLP utils
=#

@kernel function _copy_diag!(dest, src)
    i = @index(Global)
    @inbounds dest[i] = src[i, i]
end

function MadNLP.diag!(dest::CuVector{T}, src::CuMatrix{T}) where T
    @assert length(dest) == size(src, 1)
    ev = _copy_diag!(CUDABackend())(dest, src, ndrange=length(dest))
    synchronize(CUDABackend())
end

@kernel function _add_diagonal!(dest, src1, src2)
    i = @index(Global)
    @inbounds dest[i, i] = src1[i] + src2[i]
end

function MadNLP.diag_add!(dest::CuMatrix, src1::CuVector, src2::CuVector)
    ev = _add_diagonal!(CUDABackend())(dest, src1, src2, ndrange=size(dest, 1))
    synchronize(CUDABackend())
end

#=
   Contiguous views do not dispatch to the correct copyto! kernel
   in CUDA.jl. To avoid fallback to the (slow) implementation in Julia Base,
   we overload the copyto! operator locally
=#
_copyto!(dest::CuArray, src::Array) = copyto!(dest, src)
function _copyto!(dest::CuArray, src::SubArray)
    @assert src.stride1 == 1 # src array should be one-strided
    n = length(dest)
    offset = src.offset1
    p_src = parent(src)
    copyto!(dest, 1, p_src, offset+1, n)
end

#=
    MadNLP kernels
=#

# #=
#     AbstractDenseKKTSystem
# =#

# #=
#     DenseKKTSystem kernels
# =#

@kernel function _build_dense_kkt_system_kernel!(
    dest, hess, jac, pr_diag, du_diag, diag_hess, ind_ineq, n, m, ns
)
    i, j = @index(Global, NTuple)
    @inbounds if (i <= n)
        # Transfer Hessian
        if (i == j)
            dest[i, i] = pr_diag[i] + diag_hess[i]
        else
            dest[i, j] = hess[i, j]
        end
    elseif i <= n + ns
        # Transfer slack diagonal
        dest[i, i] = pr_diag[i]
        # Transfer Jacobian wrt slack
        js = i - n
        is = ind_ineq[js]
        dest[is + n + ns, is + n] = - 1
        dest[is + n, is + n + ns] = - 1
    elseif i <= n + ns + m
        # Transfer Jacobian wrt variable x
        i_ = i - n - ns
        dest[i, j] = jac[i_, j]
        dest[j, i] = jac[i_, j]
        # Transfer dual regularization
        dest[i, i] = du_diag[i_]
    end
end
 
function MadNLP._build_dense_kkt_system!(
    dest::CuMatrix, hess::CuMatrix, jac::CuMatrix,
    pr_diag::CuVector, du_diag::CuVector, diag_hess::CuVector, ind_ineq,  n, m, ns
) 
    ind_ineq_gpu = ind_ineq |> CuArray
    ndrange = (n+m+ns, n)
    ev = _build_dense_kkt_system_kernel!(CUDABackend())(
        dest, hess, jac, pr_diag, du_diag, diag_hess, ind_ineq_gpu, n, m, ns,
        ndrange=ndrange
    )
    synchronize(CUDABackend())
end


# #=
#     DenseCondensedKKTSystem
# =#
# function MadNLP.get_slack_regularization(kkt::MadNLP.DenseCondensedKKTSystem{T, VT, MT}) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
#     n, ns = MadNLP.num_variables(kkt), kkt.n_ineq
#     return view(kkt.pr_diag, n+1:n+ns) |> Array
# end


@kernel function _build_jacobian_condensed_kernel!(
    dest, jac, diag_buffer, ind_ineq,  m_ineq,
)
    i, j = @index(Global, NTuple)
    @inbounds is = ind_ineq[i]
    @inbounds dest[i, j] = jac[is, j] * sqrt(diag_buffer[i])
end
 
function MadNLP._build_ineq_jac!(
    dest::CuMatrix, jac::CuMatrix, diag_buffer::CuVector,
    ind_ineq::AbstractVector, n, m_ineq,
)
    (m_ineq == 0) && return # nothing to do if no ineq. constraints
    ind_ineq_gpu = ind_ineq |> CuArray
    ndrange = (m_ineq, n)
    ev = _build_jacobian_condensed_kernel!(CUDABackend())(
        dest, jac, diag_buffer, ind_ineq_gpu, m_ineq,
        ndrange=ndrange,
    )
    synchronize(CUDABackend())
    return
end

@kernel function _build_condensed_kkt_system_kernel!(
    dest, hess, jac, pr_diag, du_diag, ind_eq, n, m_eq,
)
    i, j = @index(Global, NTuple)

    # Transfer Hessian
    @inbounds if i <= n
        if i == j
            dest[i, i] += pr_diag[i] + hess[i, i]
        else
            dest[i, j] += hess[i, j]
        end
    elseif i <= n + m_eq
        i_ = i - n
        is = ind_eq[i_]
        # Jacobian / equality
        dest[i_ + n, j] = jac[is, j]
        dest[j, i_ + n] = jac[is, j]
        # Transfer dual regularization
        dest[i_ + n, i_ + n] = du_diag[is]
    end
end

function MadNLP._build_condensed_kkt_system!(
    dest::CuMatrix, hess::CuMatrix, jac::CuMatrix,
    pr_diag::CuVector, du_diag::CuVector, ind_eq::AbstractVector, n, m_eq,
)
    ind_eq_gpu = ind_eq |> CuArray
    ndrange = (n + m_eq, n)
    ev = _build_condensed_kkt_system_kernel!(CUDABackend())(
        dest, hess, jac, pr_diag, du_diag, ind_eq_gpu, n, m_eq,
        ndrange=ndrange,
    )
    synchronize(CUDABackend())
end

function MadNLP._set_diag!(A::CuMatrix, inds, a)
    if !isempty(inds)
        _set_diag_kernel!(CUDABackend())(
            A, inds, a;
            ndrange = length(inds)
        )
    end
    synchronize(CUDABackend())
end

@kernel function _set_diag_kernel!(
    A, inds, a
    )
    i = @index(Global)
    @inbounds begin
        index = inds[i]
        A[index,index] = a
    end
end
