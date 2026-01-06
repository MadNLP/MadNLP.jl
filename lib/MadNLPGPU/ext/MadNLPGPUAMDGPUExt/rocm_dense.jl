######################################################################
##### ROCm wrappers for DenseKKTSystem / DenseCondensedKKTSystem #####
######################################################################

#=
    MadNLP._symv!
=#

MadNLP._symv!(y, A, x::ROCVector{T}, α = one(T), β = zero(T)) where T = rocBLAS.symv!('L', T(α), A, x, T(β), y)

#=
    MadNLP._ger!
=#

MadNLP._ger!(alpha::Number, x::ROCVector{T}, y::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.ger!(alpha, x, y, A)

#=
    MadNLP._madnlp_unsafe_wrap
=#

function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift=1) where {T, VT <: ROCVector{T}}
    return view(vec,shift:shift+n-1)
end

#=
    MadNLP.diag!
=#

function MadNLP.diag!(dest::ROCVector{T}, src::ROCMatrix{T}) where {T}
    @assert length(dest) == size(src, 1)
    backend = ROCBackend()
    MadNLPGPU._copy_diag_kernel!(backend)(dest, src, ndrange = length(dest))
    
    return
end

#=
    MadNLP.diag_add!
=#

function MadNLP.diag_add!(dest::ROCMatrix, src1::ROCVector, src2::ROCVector)
    backend = ROCBackend()
    MadNLPGPU._add_diagonal_kernel!(backend)(dest, src1, src2, ndrange = size(dest, 1))
    
    return
end

#=
    MadNLP._set_diag!
=#

function MadNLP._set_diag!(A::ROCMatrix, inds, a)
    if !isempty(inds)
        backend = ROCBackend()
        MadNLPGPU._set_diag_kernel!(backend)(A, inds, a; ndrange = length(inds))
        
    end
    return
end

#=
    MadNLP._build_dense_kkt_system!
=#

function MadNLP._build_dense_kkt_system!(
    dest::ROCMatrix,
    hess::ROCMatrix,
    jac::ROCMatrix,
    pr_diag::ROCVector,
    du_diag::ROCVector,
    diag_hess::ROCVector,
    ind_ineq::AbstractVector,
    n,
    m,
    ns,
)
    ind_ineq_gpu = ROCVector(ind_ineq)
    ndrange = (n + m + ns, n)
    backend = ROCBackend()
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
    dest::ROCMatrix,
    jac::ROCMatrix,
    diag_buffer::ROCVector,
    ind_ineq::AbstractVector,
    n,
    m_ineq,
)
    (m_ineq == 0) && return # nothing to do if no ineq. constraints
    ind_ineq_gpu = ROCVector(ind_ineq)
    ndrange = (m_ineq, n)
    backend = ROCBackend()
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
    dest::ROCMatrix,
    hess::ROCMatrix,
    jac::ROCMatrix,
    pr_diag::ROCVector,
    du_diag::ROCVector,
    ind_eq::AbstractVector,
    n,
    m_eq,
)
    ind_eq_gpu = ROCVector(ind_eq)
    ndrange = (n + m_eq, n)
    backend = ROCBackend()
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
