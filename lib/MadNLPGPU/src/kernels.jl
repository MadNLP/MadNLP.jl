#=
    MadNLP utils
=#

@kernel function _copy_diag!(dest, src)
    i = @index(Global)
    dest[i] = src[i, i]
end

function MadNLP.diag!(dest::CuVector{T}, src::CuMatrix{T}) where T
    @assert length(dest) == size(src, 1)
    ev = _copy_diag!(CUDADevice())(dest, src, ndrange=length(dest))
    wait(ev)
end

@kernel function _add_diagonal!(dest, src1, src2)
    i = @index(Global)
    dest[i, i] = src1[i] + src2[i]
end

function MadNLP.diag_add!(dest::CuMatrix, src1::CuVector, src2::CuVector)
    ev = _add_diagonal!(CUDADevice())(dest, src1, src2, ndrange=size(dest, 1))
    wait(ev)
end

#=
    MadNLP kernels
=#

# Overload is_valid to avoid fallback to default is_valid, slow on GPU
MadNLP.is_valid(src::CuArray) = true

# Constraint scaling
function MadNLP.set_con_scale!(con_scale::AbstractVector, jac::CuMatrix, nlp_scaling_max_gradient)
    # Compute reduction on the GPU with built-in CUDA.jl function
    d_con_scale = maximum(abs, jac, dims=2)
    copyto!(con_scale, d_con_scale)
    con_scale .= min.(1.0, nlp_scaling_max_gradient ./ con_scale)
end

@kernel function _treat_fixed_variable_kernell!(dest, ind_fixed)
    k, j = @index(Global, NTuple)
    i = ind_fixed[k]

    if i == j
        dest[i, i] = 1.0
    else
        dest[i, j] = 0.0
        dest[j, i] = 0.0
    end
end

function MadNLP.treat_fixed_variable!(kkt::MadNLP.AbstractKKTSystem{T, MT}) where {T, MT<:CuMatrix{T}}
    length(kkt.ind_fixed) == 0 && return
    aug = kkt.aug_com
    d_ind_fixed = kkt.ind_fixed |> CuVector # TODO: allocate ind_fixed directly on the GPU
    ndrange = (length(d_ind_fixed), size(aug, 1))
    ev = _treat_fixed_variable_kernell!(CUDADevice())(aug, d_ind_fixed, ndrange=ndrange)
    wait(ev)
end

#=
    DenseKKTSystem kernels
=#

function MadNLP.mul!(y::AbstractVector, kkt::MadNLP.DenseKKTSystem{T, VT, MT}, x::AbstractVector) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    @assert length(kkt._w2) == length(x)
    # x and y can be host arrays. Copy them on the device to avoid side effect.
    copyto!(kkt._w2, x)
    mul!(kkt._w1, kkt.aug_com, kkt._w2)
    copyto!(y, kkt._w1)
end

function MadNLP.jtprod!(y::AbstractVector, kkt::MadNLP.DenseKKTSystem{T, VT, MT}, x::AbstractVector) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    # x and y can be host arrays. Copy them on the device to avoid side effect.
    #
    d_x = x |> CuArray
    d_y = y |> CuArray
    mul!(d_y, kkt.jac', d_x)
    copyto!(y, d_y)
end

function MadNLP.set_aug_diagonal!(kkt::MadNLP.DenseKKTSystem{T, VT, MT}, ips::MadNLP.Solver) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}}
    # Broadcast is not working as MadNLP array are allocated on the CPU,
    # whereas pr_diag is allocated on the GPU
    copyto!(kkt.pr_diag, ips.zl./(ips.x.-ips.xl) .+ ips.zu./(ips.xu.-ips.x))
    fill!(kkt.du_diag, 0.0)
end

@kernel function _build_dense_kkt_system_kerneq!(
    dest, hess, jac, pr_diag, du_diag, diag_hess, n, m, ns
)
    i, j = @index(Global, NTuple)
    if (i <= n)
        # Transfer Hessian
        if (i == j)
            dest[i, i] = pr_diag[i] + diag_hess[i]
        elseif j <= n
            dest[i, j] = hess[i, j]
            dest[j, i] = hess[j, i]
        end
    elseif i <= n + ns
        # Transfer slack diagonal
        dest[i, i] = pr_diag[i]
    elseif i <= n + ns + m
        # Transfer Jacobian
        i_ = i - n - ns
        dest[i, j] = jac[i_, j]
        dest[j, i] = jac[i_, j]
        # Transfer dual regularization
        dest[i, i] = du_diag[i_]
    end
end

function MadNLP._build_dense_kkt_system!(
    dest::CuMatrix, hess::CuMatrix, jac::CuMatrix,
    pr_diag::CuVector, du_diag::CuVector, diag_hess::CuVector, n, m, ns
)
    ndrange = (n+m+ns, n+ns)
    ev = _build_dense_kkt_system_kerneq!(CUDADevice())(dest, hess, jac, pr_diag, du_diag, diag_hess, n, m, ns, ndrange=ndrange)
    wait(ev)
end

