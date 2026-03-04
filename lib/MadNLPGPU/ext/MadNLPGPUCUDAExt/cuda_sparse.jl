######################################################
##### CUDA stubs for SparseCondensedKKTSystem    #####
##### (functions requiring concrete CUSPARSE types) ##
######################################################

function MadNLP.transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC,
    src::MadNLP.SparseMatrixCOO,
    map,
)
    return copyto!(view(dest.nzVal, map), src.V)
end

MadNLP.nzval(H::CUSPARSE.CuSparseMatrixCSC) = H.nzVal

function MadNLP._get_sparse_csc(dims, colptr::CuVector, rowval, nzval)
    return CUSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function MadNLP._sym_length(Jt::CUSPARSE.CuSparseMatrixCSC)
    return mapreduce(
        (x, y) -> begin
            z = x - y
            div(z^2 + z, 2)
        end,
        +,
        @view(Jt.colPtr[2:end]),
        @view(Jt.colPtr[1:end-1])
    )
end

function MadNLP.get_tril_to_full(csc::CUSPARSE.CuSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    cscind = MadNLP.SparseMatrixCSC{Int,Ti}(
        Symmetric(
            MadNLP.SparseMatrixCSC{Int,Ti}(
                size(csc)...,
                Array(csc.colPtr),
                Array(csc.rowVal),
                collect(1:MadNLP.nnz(csc)),
            ),
            :L,
        ),
    )
    return CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(
        CuArray(cscind.colptr),
        CuArray(cscind.rowval),
        CuVector{Tv}(undef, MadNLP.nnz(cscind)),
        size(csc),
    ),
    view(csc.nzVal, CuArray(cscind.nzval))
end

function MadNLP.build_condensed_aug_coord!(
    kkt::MadNLP.AbstractCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T}}
    fill!(kkt.aug_com.nzVal, zero(T))
    backend = get_backend(kkt.pr_diag)
    if length(kkt.hptr) > 0
        MadNLPGPU._transfer_hessian_kernel!(backend)(
            kkt.aug_com.nzVal,
            kkt.hptr,
            kkt.hess_com.nzVal;
            ndrange = length(kkt.hptr),
        )
        synchronize(backend)
    end
    if length(kkt.dptr) > 0
        MadNLPGPU._transfer_hessian_kernel!(backend)(
            kkt.aug_com.nzVal,
            kkt.dptr,
            kkt.pr_diag;
            ndrange = length(kkt.dptr),
        )
        synchronize(backend)
    end
    if length(kkt.ext.jptrptr) > 1 # otherwise error is thrown
        MadNLPGPU._transfer_jtsj_kernel!(backend)(
            kkt.aug_com.nzVal,
            kkt.jptr,
            kkt.ext.jptrptr,
            kkt.jt_csc.nzVal,
            kkt.diag_buffer;
            ndrange = length(kkt.ext.jptrptr) - 1,
        )
        synchronize(backend)
    end
    return
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.AbstractSparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    fill!(kkt.hess_com.nzVal, zero(T))
    backend = get_backend(kkt.pr_diag)
    if length(kkt.ext.hess_com_ptrptr) > 1
        MadNLPGPU._transfer_to_csc_kernel!(backend)(
            kkt.hess_com.nzVal,
            kkt.ext.hess_com_ptr,
            kkt.ext.hess_com_ptrptr,
            kkt.hess_raw.V;
            ndrange = length(kkt.ext.hess_com_ptrptr) - 1,
        )
        synchronize(backend)
    end
    return
end

function MadNLP.compress_jacobian!(
    kkt::MadNLP.SparseCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    fill!(kkt.jt_csc.nzVal, zero(T))
    backend = get_backend(kkt.pr_diag)
    if length(kkt.ext.jt_csc_ptrptr) > 1 # otherwise error is thrown
        MadNLPGPU._transfer_to_csc_kernel!(backend)(
            kkt.jt_csc.nzVal,
            kkt.ext.jt_csc_ptr,
            kkt.ext.jt_csc_ptrptr,
            kkt.jt_coo.V;
            ndrange = length(kkt.ext.jt_csc_ptrptr) - 1,
        )
        synchronize(backend)
    end
    return
end

function MadNLP._build_condensed_aug_symbolic_hess(
    H::CUSPARSE.CuSparseMatrixCSC{Tv,Ti},
    sym,
    sym2,
) where {Tv,Ti}
    if size(H, 2) > 0
        backend = get_backend(H.nzVal)
        MadNLPGPU._build_condensed_aug_symbolic_hess_kernel!(backend)(
            sym,
            sym2,
            H.colPtr,
            H.rowVal;
            ndrange = size(H, 2),
        )
        synchronize(backend)
    end
    return
end

function MadNLP._build_condensed_aug_symbolic_jt(
    Jt::CUSPARSE.CuSparseMatrixCSC{Tv,Ti},
    sym,
    sym2,
) where {Tv,Ti}
    if size(Jt, 2) > 0
        _offsets = map(
            (i, j) -> div((j - i)^2 + (j - i), 2),
            @view(Jt.colPtr[1:end-1]),
            @view(Jt.colPtr[2:end])
        )
        offsets = cumsum(_offsets)
        backend = get_backend(Jt.nzVal)
        MadNLPGPU._build_condensed_aug_symbolic_jt_kernel!(backend)(
            sym,
            sym2,
            Jt.colPtr,
            Jt.rowVal,
            offsets;
            ndrange = size(Jt, 2),
        )
        synchronize(backend)
    end
    return
end
