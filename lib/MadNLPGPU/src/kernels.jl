#=
    MadNLP utils
=#

@kernel function _copy_diag!(dest, src)
    i = @index(Global)
    dest[i] = src[i, i]
end

function MadNLP.diag!(dest::CuVector{T}, src::CuMatrix{T}) where T
    @assert length(dest) == size(src, 1)
    ev = _copy_diag!(CUDABackend())(dest, src, ndrange=length(dest))
    synchronize(CUDABackend())
end

@kernel function _add_diagonal!(dest, src1, src2)
    i = @index(Global)
    dest[i, i] = src1[i] + src2[i]
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

# Overload MadNLP.is_valid to avoid fallback to default is_valid, slow on GPU
MadNLP.is_valid(src::CuArray) = true


#=
    AbstractDenseKKTSystem
=#

function MadNLP.jtprod!(y::AbstractVector, kkt::MadNLP.AbstractDenseKKTSystem{T, VT, MT}, x::AbstractVector) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    # Load buffers
    m = size(kkt.jac, 1)
    nx = size(kkt.jac, 2)
    ns = length(kkt.ind_ineq)
    haskey(kkt.etc, :jac_w1) || (kkt.etc[:jac_w1] = CuVector{T}(undef, m))
    haskey(kkt.etc, :jac_w2) || (kkt.etc[:jac_w2] = CuVector{T}(undef, nx))
    haskey(kkt.etc, :jac_w3) || (kkt.etc[:jac_w3] = CuVector{T}(undef, ns))

    d_x = kkt.etc[:jac_w1]::VT
    d_yx = kkt.etc[:jac_w2]::VT
    d_ys = kkt.etc[:jac_w3]::VT

    # x and y can be host arrays. Copy them on the device to avoid side effect.
    _copyto!(d_x, x)

    # / x
    LinearAlgebra.mul!(d_yx, kkt.jac', d_x)
    copyto!(parent(y), 1, d_yx, 1, nx)

    # / s
    d_ys .= -d_x[kkt.ind_ineq] .* kkt.constraint_scaling[kkt.ind_ineq]
    copyto!(parent(y), nx+1, d_ys, 1, ns)
    return
end

function MadNLP.set_aug_diagonal!(kkt::MadNLP.AbstractDenseKKTSystem{T, VT, MT}, solver::MadNLP.MadNLPSolver) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    haskey(kkt.etc, :pr_diag_host) || (kkt.etc[:pr_diag_host] = Vector{T}(undef, length(kkt.pr_diag)))
    pr_diag_h = kkt.etc[:pr_diag_host]::Vector{T}
    x = MadNLP.full(solver.x)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)
    xl = MadNLP.full(solver.xl)
    xu = MadNLP.full(solver.xu)
    # Broadcast is not working as MadNLP array are allocated on the CPU,
    # whereas pr_diag is allocated on the GPU
    pr_diag_h .= zl./(x.-xl) .+ zu./(xu.-x)
    copyto!(kkt.pr_diag, pr_diag_h)
    fill!(kkt.du_diag, 0.0)
end

#=
    DenseKKTSystem kernels
=#

function LinearAlgebra.mul!(y::AbstractVector, kkt::MadNLP.DenseKKTSystem{T, VT, MT}, x::AbstractVector) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    # Load buffers
    haskey(kkt.etc, :hess_w1) || (kkt.etc[:hess_w1] = CuVector{T}(undef, size(kkt.aug_com, 1)))
    haskey(kkt.etc, :hess_w2) || (kkt.etc[:hess_w2] = CuVector{T}(undef, size(kkt.aug_com, 1)))

    d_x = kkt.etc[:hess_w1]::VT
    d_y = kkt.etc[:hess_w2]::VT

    # x and y can be host arrays. Copy them on the device to avoid side effect.
    copyto!(d_x, x)
    symul!(d_y, kkt.aug_com, d_x)
    copyto!(y, d_y)
end
function LinearAlgebra.mul!(y::MadNLP.ReducedKKTVector, kkt::MadNLP.DenseKKTSystem{T, VT, MT}, x::MadNLP.ReducedKKTVector) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    LinearAlgebra.mul!(MadNLP.full(y), kkt, MadNLP.full(x))
end

@kernel function _build_dense_kkt_system_kernel!(
    dest, hess, jac, pr_diag, du_diag, diag_hess, ind_ineq, con_scale, n, m, ns
)
    i, j = @index(Global, NTuple)
    if (i <= n)
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
        dest[is + n + ns, is + n] = - con_scale[is]
        dest[is + n, is + n + ns] = - con_scale[is]
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
    pr_diag::CuVector, du_diag::CuVector, diag_hess::CuVector, ind_ineq, con_scale, n, m, ns
)
    ind_ineq_gpu = ind_ineq |> CuArray
    ndrange = (n+m+ns, n)
    ev = _build_dense_kkt_system_kernel!(CUDABackend())(
        dest, hess, jac, pr_diag, du_diag, diag_hess, ind_ineq_gpu, con_scale, n, m, ns,
        ndrange=ndrange
    )
    synchronize(CUDABackend())
end


#=
    DenseCondensedKKTSystem
=#
function MadNLP.get_slack_regularization(kkt::MadNLP.DenseCondensedKKTSystem{T, VT, MT}) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    n, ns = MadNLP.num_variables(kkt), kkt.n_ineq
    return view(kkt.pr_diag, n+1:n+ns) |> Array
end
function MadNLP.get_scaling_inequalities(kkt::MadNLP.DenseCondensedKKTSystem{T, VT, MT}) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    return kkt.constraint_scaling[kkt.ind_ineq] |> Array
end

@kernel function _build_jacobian_condensed_kernel!(
    dest, jac, pr_diag, ind_ineq, con_scale, n, m_ineq,
)
    i, j = @index(Global, NTuple)
    is = ind_ineq[i]
    @inbounds dest[i, j] = jac[is, j] * sqrt(pr_diag[n+i]) / con_scale[is]
end

function MadNLP._build_ineq_jac!(
    dest::CuMatrix, jac::CuMatrix, pr_diag::CuVector,
    ind_ineq::AbstractVector, ind_fixed::AbstractVector, con_scale::CuVector, n, m_ineq,
)
    (m_ineq == 0) && return # nothing to do if no ineq. constraints
    ind_ineq_gpu = ind_ineq |> CuArray
    ndrange = (m_ineq, n)
    ev = _build_jacobian_condensed_kernel!(CUDABackend())(
        dest, jac, pr_diag, ind_ineq_gpu, con_scale, n, m_ineq,
        ndrange=ndrange,
    )
    synchronize(CUDABackend())
    # need to zero the fixed components
    dest[:, ind_fixed] .= 0.0
    return
end

@kernel function _build_condensed_kkt_system_kernel!(
    dest, hess, jac, pr_diag, du_diag, ind_eq, n, m_eq,
)
    i, j = @index(Global, NTuple)

    # Transfer Hessian
    if i <= n
        if i == j
            @inbounds dest[i, i] += pr_diag[i] + hess[i, i]
        else
            @inbounds dest[i, j] += hess[i, j]
        end
    elseif i <= n + m_eq
        i_ = i - n
        @inbounds is = ind_eq[i_]
        # Jacobian / equality
        @inbounds dest[i_ + n, j] = jac[is, j]
        @inbounds dest[j, i_ + n] = jac[is, j]
        # Transfer dual regularization
        @inbounds dest[i_ + n, i_ + n] = du_diag[is]
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

function LinearAlgebra.mul!(y::AbstractVector, kkt::MadNLP.DenseCondensedKKTSystem{T, VT, MT}, x::AbstractVector) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    if length(y) == length(x) == size(kkt.aug_com, 1)
        # Load buffers
        haskey(kkt.etc, :hess_w1) || (kkt.etc[:hess_w1] = CuVector{T}(undef, size(kkt.aug_com, 1)))
        haskey(kkt.etc, :hess_w2) || (kkt.etc[:hess_w2] = CuVector{T}(undef, size(kkt.aug_com, 1)))

        d_x = kkt.etc[:hess_w1]::VT
        d_y = kkt.etc[:hess_w2]::VT

        # Call parent() as CUDA does not dispatch on proper copyto! when passed a view
        copyto!(d_x, 1, parent(x), 1, length(x))
        symul!(d_y,  kkt.aug_com, d_x)
        copyto!(y, d_y)
    else
        # Load buffers
        haskey(kkt.etc, :hess_w3) || (kkt.etc[:hess_w3] = CuVector{T}(undef, length(x)))
        haskey(kkt.etc, :hess_w4) || (kkt.etc[:hess_w4] = CuVector{T}(undef, length(y)))

        d_x = kkt.etc[:hess_w3]::VT
        d_y = kkt.etc[:hess_w4]::VT

        # Call parent() as CUDA does not dispatch on proper copyto! when passed a view
        copyto!(d_x, 1, parent(x), 1, length(x))
        MadNLP._mul_expanded!(d_y, kkt, d_x)
        copyto!(y, d_y)
    end
end
function LinearAlgebra.mul!(y::MadNLP.ReducedKKTVector, kkt::MadNLP.DenseCondensedKKTSystem{T, VT, MT}, x::MadNLP.ReducedKKTVector) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    LinearAlgebra.mul!(MadNLP.full(y), kkt, MadNLP.full(x))
end

function MadNLP.jprod_ineq!(y::AbstractVector, kkt::MadNLP.DenseCondensedKKTSystem{T, VT, MT}, x::AbstractVector) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    # Create buffers
    haskey(kkt.etc, :jac_ineq_w1) || (kkt.etc[:jac_ineq_w1] = CuVector{T}(undef, kkt.n_ineq))
    haskey(kkt.etc, :jac_ineq_w2) || (kkt.etc[:jac_ineq_w2] = CuVector{T}(undef, size(kkt.jac_ineq, 2)))

    y_d = kkt.etc[:jac_ineq_w1]::VT
    x_d = kkt.etc[:jac_ineq_w2]::VT

    # Call parent() as CUDA does not dispatch on proper copyto! when passed a view
    copyto!(x_d, 1, parent(x), 1, length(x))
    LinearAlgebra.mul!(y_d, kkt.jac_ineq, x_d)
    copyto!(parent(y), 1, y_d, 1, length(y))
end
