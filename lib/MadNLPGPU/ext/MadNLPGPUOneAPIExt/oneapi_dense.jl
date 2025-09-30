########################################################################
##### oneAPI wrappers for DenseKKTSystem / DenseCondensedKKTSystem #####
########################################################################

#=
    MadNLP.symul!
=#

MadNLP.symul!(y, A, x::oneVector{T}, α = one(T), β = zero(T)) where T = oneMKL.symv!('L', T(α), A, x, T(β), y)

#=
    MadNLP._ger!
=#

MadNLP._ger!(alpha::Number, x::oneVector{T}, y::oneVector{T}, A::oneMatrix{T}) where T = oneMKL.ger!(alpha, x, y, A)

#=
    MadNLP._madnlp_unsafe_wrap
=#

function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift=1) where {T, VT <: oneVector{T}}
    return view(vec,shift:shift+n-1)
end

#=
    MadNLP.diag!
=#

function MadNLP.diag!(dest::oneVector{T}, src::oneMatrix{T}) where {T}
    @assert length(dest) == size(src, 1)
    backend = oneAPIBackend()
    MadNLPGPU._copy_diag_kernel!(backend)(dest, src, ndrange = length(dest))
    synchronize(backend)
    return
end

#=
    MadNLP.diag_add!
=#

function MadNLP.diag_add!(dest::oneMatrix, src1::oneVector, src2::oneVector)
    backend = oneAPIBackend()
    MadNLPGPU._add_diagonal_kernel!(backend)(dest, src1, src2, ndrange = size(dest, 1))
    synchronize(backend)
    return
end

#=
    MadNLP._set_diag!
=#

function MadNLP._set_diag!(A::oneMatrix, inds, a)
    if !isempty(inds)
        backend = oneAPIBackend()
        MadNLPGPU._set_diag_kernel!(backend)(A, inds, a; ndrange = length(inds))
        synchronize(backend)
    end
    return
end

#=
    MadNLP._build_dense_kkt_system!
=#

function MadNLP._build_dense_kkt_system!(
    dest::oneMatrix,
    hess::oneMatrix,
    jac::oneMatrix,
    pr_diag::oneVector,
    du_diag::oneVector,
    diag_hess::oneVector,
    ind_ineq::AbstractVector,
    n,
    m,
    ns,
)
    ind_ineq_gpu = oneVector(ind_ineq)
    ndrange = (n + m + ns, n)
    backend = oneAPIBackend()
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
    synchronize(backend)
    return
end

#=
    MadNLP._build_ineq_jac!
=#

function MadNLP._build_ineq_jac!(
    dest::oneMatrix,
    jac::oneMatrix,
    diag_buffer::oneVector,
    ind_ineq::AbstractVector,
    n,
    m_ineq,
)
    (m_ineq == 0) && return # nothing to do if no ineq. constraints
    ind_ineq_gpu = oneVector(ind_ineq)
    ndrange = (m_ineq, n)
    backend = oneAPIBackend()
    MadNLPGPU._build_jacobian_condensed_kernel!(backend)(
        dest,
        jac,
        diag_buffer,
        ind_ineq_gpu,
        m_ineq,
        ndrange = ndrange,
    )
    synchronize(backend)
    return
end

#=
    MadNLP._build_condensed_kkt_system!
=#

function MadNLP._build_condensed_kkt_system!(
    dest::oneMatrix,
    hess::oneMatrix,
    jac::oneMatrix,
    pr_diag::oneVector,
    du_diag::oneVector,
    ind_eq::AbstractVector,
    n,
    m_eq,
)
    ind_eq_gpu = oneVector(ind_eq)
    ndrange = (n + m_eq, n)
    backend = oneAPIBackend()
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
    synchronize(backend)
    return
end
