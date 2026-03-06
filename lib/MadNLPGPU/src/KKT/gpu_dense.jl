######################################################################
##### GPU wrappers for DenseKKTSystem / DenseCondensedKKTSystem #####
######################################################################

#=
    MadNLP._madnlp_unsafe_wrap
=#

function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift = 1) where {T, VT <: AbstractGPUVector{T}}
    return view(vec, shift:(shift + n - 1))
end

#=
    MadNLP.diag!
=#

function MadNLP.diag!(dest::AbstractGPUVector{T}, src::AbstractGPUMatrix{T}) where {T}
    @assert length(dest) == size(src, 1)
    backend = get_backend(dest)
    MadNLPGPU._copy_diag_kernel!(backend)(dest, src, ndrange = length(dest))
    return
end

#=
    MadNLP.diag_add!
=#

function MadNLP.diag_add!(dest::AbstractGPUMatrix, src1::AbstractGPUVector, src2::AbstractGPUVector)
    backend = get_backend(dest)
    MadNLPGPU._add_diagonal_kernel!(backend)(dest, src1, src2, ndrange = size(dest, 1))
    return
end

#=
    MadNLP._set_diag!
=#

function MadNLP._set_diag!(A::AbstractGPUMatrix, inds, a)
    if !isempty(inds)
        backend = get_backend(A)
        MadNLPGPU._set_diag_kernel!(backend)(A, inds, a; ndrange = length(inds))

    end
    return
end

#=
    MadNLP._build_dense_kkt_system!
=#

function MadNLP._build_dense_kkt_system!(
        dest::AbstractGPUMatrix,
        hess::AbstractGPUMatrix,
        jac::AbstractGPUMatrix,
        pr_diag::AbstractGPUVector,
        du_diag::AbstractGPUVector,
        diag_hess::AbstractGPUVector,
        ind_ineq::AbstractVector,
        n,
        m,
        ns,
    )
    backend = get_backend(dest)
    ind_ineq_gpu = adapt(backend, ind_ineq)
    ndrange = (n + m + ns, n)
    MadNLPGPU._build_dense_kkt_system_kernel!(backend)(
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
    return
end

#=
    MadNLP._build_ineq_jac!
=#

function MadNLP._build_ineq_jac!(
        dest::AbstractGPUMatrix,
        jac::AbstractGPUMatrix,
        diag_buffer::AbstractGPUVector,
        ind_ineq::AbstractVector,
        n,
        m_ineq,
    )
    (m_ineq == 0) && return # nothing to do if no ineq. constraints
    backend = get_backend(dest)
    ind_ineq_gpu = adapt(backend, ind_ineq)
    ndrange = (m_ineq, n)
    MadNLPGPU._build_jacobian_condensed_kernel!(backend)(
        dest,
        jac,
        diag_buffer,
        ind_ineq_gpu,
        m_ineq,
        ndrange = ndrange,
    )
    return
end

#=
    MadNLP._build_condensed_kkt_system!
=#

function MadNLP._build_condensed_kkt_system!(
        dest::AbstractGPUMatrix,
        hess::AbstractGPUMatrix,
        jac::AbstractGPUMatrix,
        pr_diag::AbstractGPUVector,
        du_diag::AbstractGPUVector,
        ind_eq::AbstractVector,
        n,
        m_eq,
    )
    backend = get_backend(dest)
    ind_eq_gpu = adapt(backend, ind_eq)
    ndrange = (n + m_eq, n)
    MadNLPGPU._build_condensed_kkt_system_kernel!(backend)(
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
    return
end
