function MadNLP._update_SY!(qn::MadNLP.CompactLBFGS, s::CuVector, y::CuVector)
    if qn.current_mem < qn.max_mem
        qn.current_mem += 1
        n, k = size(qn)
        vec_Sk = vec(qn.Sk)
        vec_Yk = vec(qn.Yk)
        resize!(vec_Sk, n*k)
        resize!(vec_Yk, n*k)
        qn.Sk = reshape(vec_Sk, n, k)
        qn.Yk = reshape(vec_Yk, n, k)
        view(qn.Sk, 1:n, k) .= s
        view(qn.Yk, 1:n, k) .= y
        MadNLP._resize!(qn)
    else
        n, k = size(qn)
        # Shift
        backend = CUDABackend()
        _update_SY_kernel!(backend)(k, qn.Sk, ndrange=n)
        _update_SY_kernel!(backend)(k, qn.Yk, ndrange=n)
        synchronize(backend)

        # Latest element
        view(qn.Sk, 1:n, k) .= s
        view(qn.Yk, 1:n, k) .= y
    end
end

function MadNLP._refresh_D!(qn::MadNLP.CompactLBFGS, sk::CuVector, yk::CuVector)
    k = qn.current_mem
    sTy = LinearAlgebra.dot(sk, yk)
    if length(qn.Dk) < qn.max_mem
        CUDA.@allowscalar push!(qn.Dk, sTy)
    else
        # shift
        @inbounds for i in 1:k-1
            CUDA.@allowscalar qn.Dk[i] = qn.Dk[i+1]
        end
        CUDA.@allowscalar qn.Dk[k] = sTy
    end
end

function MadNLP._refresh_L!(qn::MadNLP.CompactLBFGS{T,VT}) where {T, VT<:CuVector}
    p = size(qn.Lk, 1)
    LinearAlgebra.mul!(qn.Lk, qn.Sk', qn.Yk)
    backend = CUDABackend()
    _refresh_L_kernel!(backend)(qn.Lk, ndrange=(p, p))
    synchronize(backend)
end

function MadNLP.symmetrize!(A::CuMatrix)
    n, m = size(A)
    @assert n == m
    backend = CUDABackend()
    symmetrize_kernel!(backend)(A, ndrange=(n, n))
    synchronize(backend)
    return A
end
