#=
    GPU wrappers for DenseKKTSystem/DenseCondensedKKTSystem
=#

#=
    MadNLP.diag!
=#

@kernel function _copy_diag_kernel!(dest, src)
    i = @index(Global)
    @inbounds dest[i] = src[i, i]
end

function MadNLP.diag!(dest::CuVector{T}, src::CuMatrix{T}) where {T}
    @assert length(dest) == size(src, 1)
    _copy_diag_kernel!(CUDABackend())(dest, src, ndrange = length(dest))
    synchronize(CUDABackend())
    return
end

#=
    MadNLP.diag_add!
=#

@kernel function _add_diagonal_kernel!(dest, src1, src2)
    i = @index(Global)
    @inbounds dest[i, i] = src1[i] + src2[i]
end

function MadNLP.diag_add!(dest::CuMatrix, src1::CuVector, src2::CuVector)
    _add_diagonal_kernel!(CUDABackend())(dest, src1, src2, ndrange = size(dest, 1))
    synchronize(CUDABackend())
    return
end

#=
    MadNLP._set_diag!
=#

@kernel function _set_diag_kernel!(A, inds, a)
    i = @index(Global)
    @inbounds begin
        index = inds[i]
        A[index, index] = a
    end
end

function MadNLP._set_diag!(A::CuMatrix, inds, a)
    if !isempty(inds)
        _set_diag_kernel!(CUDABackend())(A, inds, a; ndrange = length(inds))
        synchronize(CUDABackend())
    end
    return
end

#=
    MadNLP._build_dense_kkt_system!
=#

@kernel function _build_dense_kkt_system_kernel!(
    dest,
    hess,
    jac,
    pr_diag,
    du_diag,
    diag_hess,
    ind_ineq,
    n,
    m,
    ns,
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
        dest[is+n+ns, is+n] = -1
        dest[is+n, is+n+ns] = -1
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
    dest::CuMatrix,
    hess::CuMatrix,
    jac::CuMatrix,
    pr_diag::CuVector,
    du_diag::CuVector,
    diag_hess::CuVector,
    ind_ineq,
    n,
    m,
    ns,
)
    ind_ineq_gpu = ind_ineq |> CuArray
    ndrange = (n + m + ns, n)
    _build_dense_kkt_system_kernel!(CUDABackend())(
        dest,
        hess,
        jac,
        pr_diag,
        du_diag,
        diag_hess,
        ind_ineq_gpu,
        n,
        m,
        ns,
        ndrange = ndrange,
    )
    synchronize(CUDABackend())
    return
end

#=
    MadNLP._build_ineq_jac!
=#

@kernel function _build_jacobian_condensed_kernel!(dest, jac, diag_buffer, ind_ineq, m_ineq)
    i, j = @index(Global, NTuple)
    @inbounds is = ind_ineq[i]
    @inbounds dest[i, j] = jac[is, j] * sqrt(diag_buffer[i])
end

function MadNLP._build_ineq_jac!(
    dest::CuMatrix,
    jac::CuMatrix,
    diag_buffer::CuVector,
    ind_ineq::AbstractVector,
    n,
    m_ineq,
)
    (m_ineq == 0) && return # nothing to do if no ineq. constraints
    ind_ineq_gpu = ind_ineq |> CuArray
    ndrange = (m_ineq, n)
    _build_jacobian_condensed_kernel!(CUDABackend())(
        dest,
        jac,
        diag_buffer,
        ind_ineq_gpu,
        m_ineq,
        ndrange = ndrange,
    )
    synchronize(CUDABackend())
    return
end

#=
    MadNLP._build_condensed_kkt_system!
=#

@kernel function _build_condensed_kkt_system_kernel!(
    dest,
    hess,
    jac,
    pr_diag,
    du_diag,
    ind_eq,
    n,
    m_eq,
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
        dest[i_+n, j] = jac[is, j]
        dest[j, i_+n] = jac[is, j]
        # Transfer dual regularization
        dest[i_+n, i_+n] = du_diag[is]
    end
end

function MadNLP._build_condensed_kkt_system!(
    dest::CuMatrix,
    hess::CuMatrix,
    jac::CuMatrix,
    pr_diag::CuVector,
    du_diag::CuVector,
    ind_eq::AbstractVector,
    n,
    m_eq,
)
    ind_eq_gpu = ind_eq |> CuArray
    ndrange = (n + m_eq, n)
    _build_condensed_kkt_system_kernel!(CUDABackend())(
        dest,
        hess,
        jac,
        pr_diag,
        du_diag,
        ind_eq_gpu,
        n,
        m_eq,
        ndrange = ndrange,
    )
    synchronize(CUDABackend())
    return
end

if VERSION < v"1.10"
    function MadNLP.mul_hess_blk!(
        wx::CuVector{T},
        kkt::Union{MadNLP.DenseKKTSystem,MadNLP.DenseCondensedKKTSystem},
        t,
    ) where {T}
        n = size(kkt.hess, 1)
        CUDA.CUBLAS.symv!('L', one(T), kkt.hess, @view(t[1:n]), zero(T), @view(wx[1:n]))
        fill!(@view(wx[n+1:end]), 0)
        return wx .+= t .* kkt.pr_diag
    end
end

