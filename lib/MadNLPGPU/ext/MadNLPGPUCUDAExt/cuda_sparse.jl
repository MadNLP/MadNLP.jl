######################################################
##### CUDA stubs for SparseCondensedKKTSystem    #####
##### (functions requiring concrete CUSPARSE types) ##
######################################################

function MadNLP.transfer!(
        dest::cuSPARSE.CuSparseMatrixCSC,
        src::MadNLP.SparseMatrixCOO,
        map,
    )
    return copyto!(view(dest.nzVal, map), src.V)
end

MadNLP.nzval(H::cuSPARSE.CuSparseMatrixCSC) = H.nzVal

function MadNLP._get_sparse_csc(dims, colptr::CuVector, rowval, nzval)
    return cuSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function MadNLP._sym_length(Jt::cuSPARSE.CuSparseMatrixCSC)
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

function MadNLP.get_tril_to_full(csc::cuSPARSE.CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    cscind = MadNLP.SparseMatrixCSC{Int, Ti}(
        Symmetric(
            MadNLP.SparseMatrixCSC{Int, Ti}(
                size(csc)...,
                Array(csc.colPtr),
                Array(csc.rowVal),
                collect(1:MadNLP.nnz(csc)),
            ),
            :L,
        ),
    )
    return cuSPARSE.CuSparseMatrixCSC{Tv, Ti}(
            CuArray(cscind.colptr),
            CuArray(cscind.rowval),
            CuVector{Tv}(undef, MadNLP.nnz(cscind)),
            size(csc),
        ),
        view(csc.nzVal, CuArray(cscind.nzval))
end

