#=
    Generic sparse methods
=#
function build_hessian_structure(cb::SparseCallback, ::Type{<:ExactHessian})
    hess_I = create_array(cb, Int32, cb.nnzh)
    hess_J = create_array(cb, Int32, cb.nnzh)
    _hess_sparsity_wrapper!(cb,hess_I,hess_J)
    return hess_I, hess_J
end
# NB. Quasi-Newton methods require only the sparsity pattern
#     of the diagonal term to store the term ξ I.
function build_hessian_structure(cb::SparseCallback, ::Type{<:AbstractQuasiNewton})
    hess_I = collect(Int32, 1:cb.nvar)
    hess_J = collect(Int32, 1:cb.nvar)
    return hess_I, hess_J
end

function jtprod!(y::AbstractVector, kkt::AbstractSparseKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end

get_jacobian(kkt::AbstractSparseKKTSystem) = kkt.jac_callback

nnz_jacobian(kkt::AbstractSparseKKTSystem) = nnz(kkt.jac_raw)

function compress_jacobian!(kkt::AbstractSparseKKTSystem)
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
end

function compress_jacobian!(kkt::AbstractSparseKKTSystem{T, VT, MT}) where {T, VT, MT<:Matrix{T}}
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    copyto!(kkt.jac_com, kkt.jac_raw)
end

function compress_hessian!(kkt::AbstractSparseKKTSystem)
    transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

#--------------------------------------------------------------------------------------------%

function jtprod!(y::AbstractVector, kkt::AbstractScaledSparseKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end

get_jacobian(kkt::AbstractScaledSparseKKTSystem) = kkt.jac_callback

nnz_jacobian(kkt::AbstractScaledSparseKKTSystem) = nnz(kkt.jac_raw)

function compress_jacobian!(kkt::AbstractScaledSparseKKTSystem)
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
end

function compress_jacobian!(kkt::AbstractScaledSparseKKTSystem{T, VT, MT}) where {T, VT, MT<:Matrix{T}}
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    copyto!(kkt.jac_com, kkt.jac_raw)
end

function compress_hessian!(kkt::AbstractScaledSparseKKTSystem)
    transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

#-------------------------------------------------------------------------------------#

get_sparse_condensed_ext(::Type{Vector{T}},args...) where T = nothing

function _sym_length(Jt)
    len = 0
    for i=1:size(Jt,2)
        n = Jt.colptr[i+1] - Jt.colptr[i]
        len += div(n^2 + n, 2)
    end
    return len
end

function _build_condensed_aug_symbolic_hess(H, sym, sym2)
    for i in 1:size(H,2)
        for j in H.colptr[i]:H.colptr[i+1]-1
            c = H.rowval[j]
            sym[j] = (0,j,0)
            sym2[j] = (c,i)
        end
    end
end

function _build_condensed_aug_symbolic_jt(Jt, sym, sym2)

    cnt = 0
    for i in 1:size(Jt,2)
        for j in Jt.colptr[i]:Jt.colptr[i+1]-1
            for k in j:Jt.colptr[i+1]-1
                c1 = Jt.rowval[j]
                c2 = Jt.rowval[k]
                sym[cnt+=1] = (i,j,k)
                sym2[cnt] = (c2,c1)
            end
        end
    end
end

function getptr(array; by = (x,y)->x != y)
    bitarray = similar(array, Bool, length(array)+1)
    fill!(bitarray, true)
    bitarray[2:end-1] .= by.(@view(array[1:end-1]),  @view(array[2:end]))
    findall(bitarray)
end

nzval(H) = H.nzval

function _get_sparse_csc(dims, colptr, rowval, nzval)
    SparseMatrixCSC(
        dims...,
        colptr,
        rowval,
        nzval
    )
end

function _first_and_last_col(sym2,ptr2)
    first= sym2[1][2]
    last = sym2[ptr2[end]][2]
    return (first, last)
end

function _set_colptr!(colptr, ptr2, sym2, guide)
    for i in @view(ptr2[2:end])

        (~,prevcol) = sym2[i-1]
        (row,col) = sym2[i]

        fill!(@view(colptr[prevcol+1:col]), guide[i])
    end
end

function _build_condensed_aug_coord!(aug_com::SparseMatrixCSC{Tv,Ti}, pr_diag, H, Jt, diag_buffer, dptr, hptr, jptr) where {Tv, Ti}
    fill!(aug_com.nzval, zero(Tv))

    @simd for idx in eachindex(hptr)
        i,j = hptr[idx]
        aug_com.nzval[i] += H.nzval[j]
    end

    @simd for idx in eachindex(dptr)
        i,j = dptr[idx]
        aug_com.nzval[i] += pr_diag[j]
    end

    @simd for idx in eachindex(jptr)
        (i,(j,k,l)) = jptr[idx]
        aug_com.nzval[i] += diag_buffer[j] * Jt.nzval[k] * Jt.nzval[l]
    end
end

function build_condensed_aug_coord!(kkt::AbstractCondensedKKTSystem{T,VT,MT}) where {T, VT, MT <: SparseMatrixCSC{T}}
    _build_condensed_aug_coord!(
        kkt.aug_com, kkt.pr_diag, kkt.hess_com, kkt.jt_csc, kkt.diag_buffer,
        kkt.dptr, kkt.hptr, kkt.jptr
    )
end

function build_condensed_aug_symbolic(H::AbstractSparseMatrix{Tv,Ti}, Jt) where {Tv, Ti}
    nnzjtsj = _sym_length(Jt)

    sym = similar(nzval(H), Tuple{Int,Int,Int},
        size(H,2) + nnz(H) + nnzjtsj
    )
    sym2 = similar(nzval(H), Tuple{Int,Int},
        size(H,2) + nnz(H) + nnzjtsj
    )
    dptr = similar(nzval(H), Tuple{Ti,Ti},
        size(H,2)
    )
    hptr = similar(nzval(H), Tuple{Ti,Ti},
        nnz(H)
    )
    jptr = similar(nzval(H), Tuple{Ti,Tuple{Ti,Ti,Ti}},
        nnzjtsj
    )
    colptr = fill!(
        similar(nzval(H), Ti, size(H,1)+1),
        one(Tv)
    )

    n = size(H,2)

    map!(
        i->(-1,i,0),
        @view(sym[1:n]),
        1:size(H,2)
    )
    map!(
        i->(i,i),
        @view(sym2[1:n]),
        1:size(H,2)
    )

    _build_condensed_aug_symbolic_hess(
        H,
        @view(sym[n+1:n+nnz(H)]),
        @view(sym2[n+1:n+nnz(H)])
    )
    _build_condensed_aug_symbolic_jt(
        Jt,
        @view(sym[n+nnz(H)+1:n+nnz(H) + nnzjtsj]),
        @view(sym2[n+nnz(H)+1:n+nnz(H)+nnzjtsj])
    )

    p = sortperm(sym2; by = ((i,j),) -> (j,i))
    permute!(sym, p)
    permute!(sym2, p)

    by(x,y) = x != y

    bitarray = similar(sym2, Bool, length(sym2))
    fill!(bitarray, true)
    bitarray[2:end] .= by.(@view(sym2[1:end-1]),  @view(sym2[2:end]))
    guide = cumsum(bitarray)

    b = findall(x->x[1] == -1, sym)
    dptr = map((x,y)->(Int32(x),Int32(y[2])), @view(guide[b]), @view(sym[b]))

    b = findall(x->x[1] == 0, sym)
    hptr = map((x,y)->(Int32(x),Int32(y[2])), @view(guide[b]), @view(sym[b]))

    b = findall(x->x[1] != -1 && x[1] != 0, sym)
    jptr = map((x,y)->(Int32(x),y), @view(guide[b]), @view(sym[b]))


    ptr = findall(bitarray)
    rowval = map(((row,col),)->Int32(row), @view(sym2[ptr]))

    by2(x,y) = x[2] != y[2]
    bitarray[2:end] .= by2.(@view(sym2[1:end-1]),  @view(sym2[2:end]))
    ptr2 = findall(bitarray)

    first, last = _first_and_last_col(sym2,ptr2)

    fill!(
        @view(colptr[1:first]),
        1
    )

    _set_colptr!(colptr, ptr2, sym2, guide)

    fill!(
        @view(colptr[last+1:end]),
        length(ptr)+1
    )

    aug_com = _get_sparse_csc(
        size(H),
        colptr,
        rowval,
        similar(nzval(H), length(ptr))
    )

    return aug_com, dptr, hptr, jptr
end
