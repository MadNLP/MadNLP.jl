######################################################################
##### CUDA wrappers for DenseKKTSystem / DenseCondensedKKTSystem #####
######################################################################

#=
    MadNLP._symv!
=#

MadNLP._symv!(y, A, x::CuVector{T}, α = one(T), β = zero(T)) where T = CUBLAS.symv!('L', T(α), A, x, T(β), y)

#=
    MadNLP._ger!
=#

MadNLP._ger!(alpha::Number, x::CuVector{T}, y::CuVector{T}, A::CuMatrix{T}) where T = CUBLAS.ger!(alpha, x, y, A)

#=
    MadNLP._madnlp_unsafe_wrap
=#

function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift=1) where {T, VT <: CuVector{T}}
    return view(vec,shift:shift+n-1)
end

#=
    MadNLP.diag!
=#

function MadNLP.diag!(dest::CuVector{T}, src::CuMatrix{T}) where {T}
    @assert length(dest) == size(src, 1)
    backend = CUDABackend()
    MadNLPGPU._copy_diag_kernel!(backend)(dest, src, ndrange = length(dest))
    return
end

#=
    MadNLP.diag_add!
=#

function MadNLP.diag_add!(dest::CuMatrix, src1::CuVector, src2::CuVector)
    backend = CUDABackend()
    MadNLPGPU._add_diagonal_kernel!(backend)(dest, src1, src2, ndrange = size(dest, 1))
    return
end

#=
    MadNLP._set_diag!
=#

function MadNLP._set_diag!(A::CuMatrix, inds, a)
    if !isempty(inds)
        backend = CUDABackend()
        MadNLPGPU._set_diag_kernel!(backend)(A, inds, a; ndrange = length(inds))
    end
    return
end

#=
    MadNLP._build_dense_kkt_system!
=#

function MadNLP._build_dense_kkt_system!(
    dest::CuMatrix,
    hess::CuMatrix,
    jac::CuMatrix,
    pr_diag::CuVector,
    du_diag::CuVector,
    diag_hess::CuVector,
    ind_ineq::AbstractVector,
    n,
    m,
    ns,
)
    ind_ineq_gpu = CuVector(ind_ineq)
    ndrange = (n + m + ns, n)
    backend = CUDABackend()
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
    dest::CuMatrix,
    jac::CuMatrix,
    diag_buffer::CuVector,
    ind_ineq::AbstractVector,
    n,
    m_ineq,
)
    (m_ineq == 0) && return # nothing to do if no ineq. constraints
    ind_ineq_gpu = CuVector(ind_ineq)
    ndrange = (m_ineq, n)
    backend = CUDABackend()
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
    dest::CuMatrix,
    hess::CuMatrix,
    jac::CuMatrix,
    pr_diag::CuVector,
    du_diag::CuVector,
    ind_eq::AbstractVector,
    n,
    m_eq,
)
    ind_eq_gpu = CuVector(ind_eq)
    ndrange = (n + m_eq, n)
    backend = CUDABackend()
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
