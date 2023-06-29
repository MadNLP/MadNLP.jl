function CuMadNLPSolver(nlp::AbstractNLPModel{T}; kwargs...) where T
    opt_ipm, opt_linear_solver, logger = MadNLP.load_options(; linear_solver=LapackGPUSolver, kwargs...)

    @assert is_supported(opt_ipm.linear_solver, T)
    MT = CuMatrix{T}
    VT = CuVector{T}
    # Determine Hessian approximation
    QN = if opt_ipm.hessian_approximation == MadNLP.DENSE_BFGS
        MadNLP.BFGS{T, VT}
    elseif opt_ipm.hessian_approximation == MadNLP.DENSE_DAMPED_BFGS
        MadNLP.DampedBFGS{T, VT}
    else
        MadNLP.ExactHessian{T, VT}
    end
    KKTSystem = if (opt_ipm.kkt_system == MadNLP.SPARSE_KKT_SYSTEM) || (opt_ipm.kkt_system == MadNLP.SPARSE_UNREDUCED_KKT_SYSTEM)
        error("Sparse KKT system are currently not supported on CUDA GPU.\n" *
            "Please use `DENSE_KKT_SYSTEM` or `DENSE_CONDENSED_KKT_SYSTEM` instead.")
    elseif opt_ipm.kkt_system == MadNLP.DENSE_KKT_SYSTEM
        MadNLP.DenseKKTSystem{T, VT, MT, QN}
    elseif opt_ipm.kkt_system == MadNLP.DENSE_CONDENSED_KKT_SYSTEM
        MadNLP.DenseCondensedKKTSystem{T, VT, MT, QN}
    end
    return MadNLP.MadNLPSolver{T,KKTSystem}(nlp, opt_ipm, opt_linear_solver; logger=logger)
end

function MadNLP.coo_to_csc(coo::MadNLP.SparseMatrixCOO{T,I,VT,VI}) where {T,I, VT <: CuArray, VI <: CuArray}
    csc, map = MadNLP.coo_to_csc(
        MadNLP.SparseMatrixCOO(coo.m, coo.n, Array(coo.I), Array(coo.J), Array(coo.V))
    )

    return CUDA.CUSPARSE.CuSparseMatrixCSC(csc), CuArray(map) 
end

function MadNLP.get_tril_to_full(csc::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    cscind = MadNLP.SparseMatrixCSC{Int,Ti}(
        MadNLP.Symmetric(
            MadNLP.SparseMatrixCSC{Int,Ti}(
                size(csc)...,
                Array(csc.colPtr),
                Array(csc.rowVal),
                collect(1:MadNLP.nnz(csc))
            ),
            :L
        )
    )
    return CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(
        CuArray(cscind.colptr),
        CuArray(cscind.rowval),
        CuVector{Tv}(undef,MadNLP.nnz(cscind)),
        size(csc),
    ),
    view(csc.nzVal,CuArray(cscind.nzval))
end



function MadNLP.transfer!(dest::CUDA.CUSPARSE.CuSparseMatrixCSC, src::MadNLP.SparseMatrixCOO, map)
    copyto!(view(dest.nzVal, map), src.V)
end

function MadNLP.get_con_scale(jac_I,jac_buffer::VT, ncon, nnzj, max_gradient) where {T, VT <: CuVector{T}}
    con_scale_cpu, jac_scale_cpu = MadNLP.get_con_scale(
        Array(jac_I),
        Array(jac_buffer),
        ncon, nnzj,
        max_gradient
    )
    return CuArray(con_scale_cpu), CuArray(jac_scale_cpu)
end

function MadNLP.build_condensed_aug_symbolic(
    hess_com::CUDA.CUSPARSE.CuSparseMatrixCSC,
    jt_csc
    )
    aug_com, dptr, hptr, jptr = MadNLP.build_condensed_aug_symbolic(
        MadNLP.SparseMatrixCSC(hess_com),
        MadNLP.SparseMatrixCSC(jt_csc)
    )

    return CUDA.CUSPARSE.CuSparseMatrixCSC(aug_com), CuArray(dptr), CuArray(hptr), CuArray(jptr)
end

function MadNLP.build_condensed_aug_coord!(kkt::MadNLP.SparseCondensedKKTSystem{T,VT,MT}) where {T, VT, MT <: CUDA.CUSPARSE.CuSparseMatrixCSC{T}}
    fill!(kkt.aug_com.nzVal, zero(T))
    _transfer!(CUDABackend())(kkt.aug_com.nzVal, kkt.hptr, kkt.hess_com.nzVal; ndrange = length(kkt.hptr))
    synchronize(CUDABackend())
    _transfer!(CUDABackend())(kkt.aug_com.nzVal, kkt.dptr, kkt.pr_diag; ndrange = length(kkt.dptr))
    synchronize(CUDABackend())
    _jtsj!(CUDABackend())(kkt.aug_com.nzVal, kkt.jptr, kkt.ext.jptrptr, kkt.jt_csc.nzVal, kkt.diag_buffer; ndrange = length(kkt.ext.jptrptr)-1)
    synchronize(CUDABackend())
end

@kernel function _transfer!(y, @Const(ptr), @Const(x))
    index = @index(Global)
    i,j = ptr[index]
    y[i] += x[j]
end

@kernel function _jtsj!(y, @Const(ptr), @Const(ptrptr), @Const(x), @Const(s))
    index = @index(Global)
    for index2 in ptrptr[index]:ptrptr[index+1]-1
        i,(j,k,l) = ptr[index2]
        y[i] += s[j] * x[k] * x[l]
    end
end


function MadNLP.get_sparse_condensed_ext(
    ::Type{VT},
    jptr, jt_map, hess_map, 
    ) where {T, VT <: CuVector{T}}

    hess_com_ptr = sort!([(j,i) for (i,j) in enumerate(Array(hess_map))])
    jt_csc_ptr = sort!([(j,i) for (i,j) in enumerate(Array(jt_map))])

    jptrptr = getptr(Array(jptr))
    hess_com_ptrptr = getptr(hess_com_ptr)
    jt_csc_ptrptr = getptr(jt_csc_ptr)
    
    return (
        jptrptr = CuArray(jptrptr),
        hess_com_ptr = CuArray(hess_com_ptr),
        hess_com_ptrptr = CuArray(hess_com_ptrptr),
        jt_csc_ptr = CuArray(jt_csc_ptr),
        jt_csc_ptrptr = CuArray(jt_csc_ptrptr), 
    )
end

function getptr(arr)
    ptr = similar(arr, Int, length(arr)+1)
    prev = 0
    cnt = 0
    for i=1:length(arr)
        cur = arr[i][1]
        if prev != cur
            ptr[cnt += 1] = i
            prev = cur
        end
    end
    ptr[cnt+=1] = length(arr)+1
    
    return resize!(ptr, cnt)
end


function MadNLP.mul!(y::CuVector{Tv}, A::MadNLP.Symmetric{Tv, CUDA.CUSPARSE.CuSparseMatrixCSC{Tv, Ti}}, x::CuVector{Tv}, a::Number, b::Number) where {Tv, Ti}
    m, n = size(A)
    MadNLP.mul!(y, A.data , x, a, b )
    MadNLP.mul!(y, A.data', x, a, 1.)
    _mul!(CUDABackend())(y, m, A.data.nnz, A.data.colPtr, A.data.rowVal, A.data.nzVal, x, a; ndrange = n)
    synchronize(CUDABackend())

    return y
end

@kernel function _mul!(y, @Const(m), @Const(nnz), @Const(colptr), @Const(rowval), @Const(nzval), @Const(x), @Const(a))
    
    col = @index(Global)
    ptr = colptr[col]
    if (ptr != colptr[col+1]) && (ptr <= nnz)
        if rowval[ptr] == col
            y[col] -= a * nzval[ptr] * x[col]
        end
    end
end

function MadNLP.initialize!(kkt::MadNLP.AbstractSparseKKTSystem{T,VT}) where {T, VT <: CuVector{T}}
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    fill!(kkt.hess_com.nzVal, 0.) # so that mul! in the initial primal-dual solve has no effect
end

function MadNLP.compress_hessian!(kkt::MadNLP.AbstractSparseKKTSystem{T, VT, MT}) where {T, VT, MT<:CUDA.CUSPARSE.CuSparseMatrixCSC{T, Int32}}
    fill!(kkt.hess_com.nzVal, zero(T))
    _transfer!(CUDABackend())(kkt.hess_com.nzVal, kkt.ext.hess_com_ptr, kkt.ext.hess_com_ptrptr, kkt.hess_raw.V; ndrange = length(kkt.ext.hess_com_ptrptr)-1)
    synchronize(CUDABackend())
end
function MadNLP.compress_jacobian!(kkt::MadNLP.SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:CUDA.CUSOLVER.CuSparseMatrixCSC{T, Int32}}
    fill!(kkt.jt_csc.nzVal, zero(T))
    _transfer!(CUDABackend())(kkt.jt_csc.nzVal, kkt.ext.jt_csc_ptr, kkt.ext.jt_csc_ptrptr, kkt.jt_coo.V; ndrange = length(kkt.ext.jt_csc_ptrptr)-1)
    synchronize(CUDABackend())    
end

@kernel function _transfer!(y, @Const(ptr), @Const(ptrptr), @Const(x))
    index = @index(Global)
    for index2 in ptrptr[index]:ptrptr[index+1]-1
        i,j = ptr[index2]
        y[i] += x[j]
    end
end

