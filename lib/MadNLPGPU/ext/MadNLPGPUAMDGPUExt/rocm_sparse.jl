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
        @view(Jt.colPtr[1:(end - 1)])
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

