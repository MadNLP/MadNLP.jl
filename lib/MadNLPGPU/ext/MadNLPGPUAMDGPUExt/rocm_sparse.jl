######################################################
##### ROCm stubs for SparseCondensedKKTSystem    #####
##### (functions requiring concrete rocSPARSE types) #
######################################################

function MadNLP.transfer!(
    dest::rocSPARSE.ROCSparseMatrixCSC,
    src::MadNLP.SparseMatrixCOO,
    map,
)
    return copyto!(view(dest.nzVal, map), src.V)
end

MadNLP.nzval(H::rocSPARSE.ROCSparseMatrixCSC) = H.nzVal

function MadNLP._get_sparse_csc(dims, colptr::ROCVector, rowval, nzval)
    return rocSPARSE.ROCSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function MadNLP._sym_length(Jt::rocSPARSE.ROCSparseMatrixCSC)
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

function MadNLP.get_tril_to_full(csc::rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
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
    return rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(
        ROCArray(cscind.colptr),
        ROCArray(cscind.rowval),
        ROCVector{Tv}(undef, MadNLP.nnz(cscind)),
        size(csc),
    ),
    view(csc.nzVal, ROCArray(cscind.nzval))
end

function MadNLP.build_condensed_aug_coord!(
    kkt::MadNLP.AbstractCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:rocSPARSE.ROCSparseMatrixCSC{T}}
    fill!(kkt.aug_com.nzVal, zero(T))
    backend = get_backend(kkt.pr_diag)
    if length(kkt.hptr) > 0
        MadNLPGPU._transfer_hessian_kernel!(backend)(
            kkt.aug_com.nzVal,
            kkt.hptr,
            kkt.hess_com.nzVal;
            ndrange = length(kkt.hptr),
        )
    end
    if length(kkt.dptr) > 0
        MadNLPGPU._transfer_hessian_kernel!(backend)(
            kkt.aug_com.nzVal,
            kkt.dptr,
            kkt.pr_diag;
            ndrange = length(kkt.dptr),
        )
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
    end
    return
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.AbstractSparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:rocSPARSE.ROCSparseMatrixCSC{T,Int32}}
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
    end
    return
end

function MadNLP.compress_jacobian!(
    kkt::MadNLP.SparseCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:rocSPARSE.ROCSparseMatrixCSC{T,Int32}}
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
    end
    return
end

function MadNLP._build_condensed_aug_symbolic_hess(
    H::rocSPARSE.ROCSparseMatrixCSC{Tv,Ti},
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
    end
    return
end

function MadNLP._build_condensed_aug_symbolic_jt(
    Jt::rocSPARSE.ROCSparseMatrixCSC{Tv,Ti},
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
    end
    return
end
