function MadNLP.coo_to_csc(coo::MadNLP.SparseMatrixCOO{T,I,VT,VI}) where {T,I, VT <: CuArray, VI <: CuArray}
    coord = map(
        (i,j,k)->((i,j),k),
        coo.I, coo.J, 1:length(coo.I)
    )
    sort!(coord, lt = (((i, j), k), ((n, m), l)) -> (j,i) < (m,n))
    
    mapptr = getptr(CUDABackend(), coord)

    colptr = similar(coo.I, size(coo,2)+1)
    
    coord_csc = coord[@view(mapptr[1:end-1])]

    ker_colptr!(CUDABackend())(colptr, coord_csc, ndrange = length(coord_csc))
    rowval = map(x -> x[1][1], coord_csc)
    nzval = similar(rowval, T)

    csc = CUDA.CUSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, size(coo))
    
    
    cscmap = similar(coo.I, Int)
    ker_index(CUDABackend())(cscmap, mapptr, coord, ndrange = length(mapptr)-1)
    
    synchronize(CUDABackend())
    
    return csc, cscmap

end

@kernel function ker_colptr!(colptr, coord)
    index = @index(Global)
    if index == 1
        ((i2,j2),k2) = coord[index]
        colptr[1:j2] .= 1
    else
        ((i1,j1),k1) = coord[index-1]
        ((i2,j2),k2) = coord[index]
        if j1 != j2
            colptr[j1+1:j2] .= index 
        end
        if index == length(coord)
            colptr[j2+1:end] .= index+1
        end
    end
    
end

@kernel function ker_index(cscmap, mapptr, coord)
    index = @index(Global)
    for l in mapptr[index]:mapptr[index+1]-1
        ((i,j),k) = coord[l]
        cscmap[k] = index
    end        
end

function getptr(backend, array; cmp = isequal)

    bitarray = similar(array, Bool, length(array) + 1)
    kergetptr(backend)(cmp, bitarray, array; ndrange = length(array) + 1)
    synchronize(backend)

    return findall(identity, bitarray)
end
@kernel function kergetptr(cmp, bitarray, @Const(array))
    I = @index(Global)
    @inbounds if I == 1
        bitarray[I] = true
    elseif I == length(array) + 1
        bitarray[I] = true
    else
        i0, j0 = array[I-1]
        i1, j1 = array[I]

        if !cmp(i0, i1)
            bitarray[I] = true
        else
            bitarray[I] = false
        end
    end
end

function CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(A::MadNLP.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}(
        CuArray(A.colptr),
        CuArray(A.rowval),
        CuArray(A.nzval),
        size(A),
    )
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

function MadNLP.build_condensed_aug_coord!(kkt::MadNLP.AbstractCondensedKKTSystem{T,VT,MT}) where {T, VT, MT <: CUDA.CUSPARSE.CuSparseMatrixCSC{T}}
    fill!(kkt.aug_com.nzVal, zero(T))
    _transfer!(CUDABackend())(kkt.aug_com.nzVal, kkt.hptr, kkt.hess_com.nzVal; ndrange = length(kkt.hptr))
    synchronize(CUDABackend())
    _transfer!(CUDABackend())(kkt.aug_com.nzVal, kkt.dptr, kkt.pr_diag; ndrange = length(kkt.dptr))
    synchronize(CUDABackend())
    if length(kkt.ext.jptrptr) > 1 # otherwise error is thrown
        _jtsj!(CUDABackend())(kkt.aug_com.nzVal, kkt.jptr, kkt.ext.jptrptr, kkt.jt_csc.nzVal, kkt.diag_buffer; ndrange = length(kkt.ext.jptrptr)-1)
    end
    synchronize(CUDABackend())
end

@kernel function _transfer!(y, @Const(ptr), @Const(x))
    index = @index(Global)
    @inbounds i,j = ptr[index]
    @inbounds y[i] += x[j]
end

@kernel function _jtsj!(y, @Const(ptr), @Const(ptrptr), @Const(x), @Const(s))
    index = @index(Global)
    @inbounds for index2 in ptrptr[index]:ptrptr[index+1]-1
        i,(j,k,l) = ptr[index2]
        y[i] += s[j] * x[k] * x[l]
    end
end


function MadNLP.get_sparse_condensed_ext(
    ::Type{VT},
    hess_com, jptr, jt_map, hess_map,
    ) where {T, VT <: CuVector{T}}

    hess_com_ptr = map((i,j)->(i,j), hess_map, 1:length(hess_map))
    if length(hess_map) > 0 # otherwise error is thrown
        sort!(hess_com_ptr)
    end

    jt_csc_ptr = map((i,j)->(i,j), jt_map, 1:length(jt_map))
    if length(jt_map) > 0 # otherwise error is thrown
        sort!(jt_csc_ptr)
    end

    by = (i,j) -> i[1] != j[1]
    jptrptr = MadNLP.getptr(jptr, by = by)
    hess_com_ptrptr = MadNLP.getptr(hess_com_ptr, by = by)
    jt_csc_ptrptr = MadNLP.getptr(jt_csc_ptr, by = by)

    diag_map_to, diag_map_fr = get_diagonal_mapping(hess_com.colPtr, hess_com.rowVal)

    return (
        jptrptr = jptrptr,
        hess_com_ptr = hess_com_ptr,
        hess_com_ptrptr = hess_com_ptrptr,
        jt_csc_ptr = jt_csc_ptr,
        jt_csc_ptrptr = jt_csc_ptrptr,
        diag_map_to = diag_map_to,
        diag_map_fr = diag_map_fr,
    )
end

function MadNLP.mul!(
    w::MadNLP.AbstractKKTVector{T,VT},
    kkt::MadNLP.SparseCondensedKKTSystem,
    x::MadNLP.AbstractKKTVector,
    alpha = one(T), beta = zero(T)
    ) where {T, VT <: CuVector{T}}


    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    # Decompose results
    xx = view(MadNLP.full(x), 1:n)
    xs = view(MadNLP.full(x), n+1:n+m)
    xz = view(MadNLP.full(x), n+m+1:n+2*m)

    # Decompose buffers
    wx = view(MadNLP.full(w), 1:n)
    ws = view(MadNLP.full(w), n+1:n+m)
    wz = view(MadNLP.full(w), n+m+1:n+2*m)

    MadNLP.mul!(wx, kkt.hess_com , xx, alpha, beta)
    MadNLP.mul!(wx, kkt.hess_com', xx, alpha, one(T))
    MadNLP.mul!(wx, kkt.jt_csc,  xz, alpha, beta)
    diag_operation(CUDABackend())(
        wx, kkt.hess_com.nzVal, xx, alpha,
        kkt.ext.diag_map_to,
        kkt.ext.diag_map_fr;
        ndrange = length(kkt.ext.diag_map_to)
    )
    synchronize(CUDABackend())

    MadNLP.mul!(wz, kkt.jt_csc', xx, alpha, one(T))
    MadNLP.axpy!(-alpha, xz, ws)
    MadNLP.axpy!(-alpha, xs, wz)

    MadNLP._kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)

end
@kernel function diag_operation(y,@Const(A),@Const(x),@Const(alpha),@Const(idx_to),@Const(idx_fr))
    i = @index(Global)
    @inbounds begin
        to = idx_to[i]
        fr = idx_fr[i]
        y[to] -= alpha * A[fr] * x[to]
    end
end

function MadNLP.mul_hess_blk!(
    wx::VT,
    kkt::Union{MadNLP.SparseKKTSystem,MadNLP.SparseCondensedKKTSystem},
    t
    ) where {T, VT <: CuVector{T}}

    n = size(kkt.hess_com, 1)
    wxx = @view(wx[1:n])
    tx  = @view(t[1:n])

    MadNLP.mul!(wxx, kkt.hess_com , tx, one(T), zero(T))
    MadNLP.mul!(wxx, kkt.hess_com', tx, one(T), one(T))
    diag_operation(CUDABackend())(
        wxx, kkt.hess_com.nzVal, tx, one(T),
        kkt.ext.diag_map_to,
        kkt.ext.diag_map_fr;
        ndrange = length(kkt.ext.diag_map_to)
    )
    synchronize(CUDABackend())

    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end


function get_diagonal_mapping(colptr, rowval)

    nnz = length(rowval)
    inds1 = findall(map((x,y)-> ((x <= nnz) && (x != y)), @view(colptr[1:end-1]), @view(colptr[2:end])))
    ptrs = colptr[inds1]
    rows = rowval[ptrs]
    inds2 = findall(inds1 .== rows)

    return rows[inds2], ptrs[inds2]
end

function MadNLP.initialize!(kkt::MadNLP.AbstractSparseKKTSystem{T,VT}) where {T, VT <: CuVector{T}}
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(kkt.hess_com.nzVal, 0.) # so that mul! in the initial primal-dual solve has no effect
end

function MadNLP.compress_hessian!(kkt::MadNLP.AbstractSparseKKTSystem{T, VT, MT}) where {T, VT, MT<:CUDA.CUSPARSE.CuSparseMatrixCSC{T, Int32}}
    fill!(kkt.hess_com.nzVal, zero(T))
    _transfer!(CUDABackend())(kkt.hess_com.nzVal, kkt.ext.hess_com_ptr, kkt.ext.hess_com_ptrptr, kkt.hess_raw.V; ndrange = length(kkt.ext.hess_com_ptrptr)-1)
    synchronize(CUDABackend())
end
function MadNLP.compress_jacobian!(kkt::MadNLP.SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:CUDA.CUSOLVER.CuSparseMatrixCSC{T, Int32}}
    fill!(kkt.jt_csc.nzVal, zero(T))
    if length(kkt.ext.jt_csc_ptrptr) > 1 # otherwise error is thrown
        _transfer!(CUDABackend())(kkt.jt_csc.nzVal, kkt.ext.jt_csc_ptr, kkt.ext.jt_csc_ptrptr, kkt.jt_coo.V; ndrange = length(kkt.ext.jt_csc_ptrptr)-1)
    end
    synchronize(CUDABackend())
end

@kernel function _transfer!(y, @Const(ptr), @Const(ptrptr), @Const(x))
    index = @index(Global)
    @inbounds for index2 in ptrptr[index]:ptrptr[index+1]-1
        i,j = ptr[index2]
        y[i] += x[j]
    end
end

function MadNLP._set_con_scale_sparse!(con_scale::VT, jac_I, jac_buffer) where {T, VT <: CuVector{T}}
    con_scale_cpu = Array(con_scale)
    MadNLP._set_con_scale_sparse!(con_scale_cpu, Array(jac_I), Array(jac_buffer))
    copyto!(con_scale, con_scale_cpu)
    # _ker_set_con_scale_sparse!(CUDABackend())(con_scale, jac_I, jac_buffer; ndrange = length(jac_I))
    # synchronize(CUDABackend())
end

# TODO fix bug
# @kernel function _ker_set_con_scale_sparse!(con_scale, jac_I, jac_buffer)
#     i = @index(Global)
#     row = jac_I[i]
#     con_scale[row] = max(con_scale[row], abs(jac_buffer[i]))
# end

function MadNLP._sym_length(Jt::CUDA.CUSPARSE.CuSparseMatrixCSC)
    return mapreduce(
        (x,y) -> begin
            z = x-y
            div(z^2 + z, 2)
        end,
        +,
        @view(Jt.colPtr[2:end]),
        @view(Jt.colPtr[1:end-1])
    )
end

function MadNLP._build_condensed_aug_symbolic_hess(H::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}, sym, sym2) where {Tv,Ti}
    ker_build_condensed_aug_symbolic_hess(CUDABackend())(
        sym, sym2, H.colPtr, H.rowVal;
        ndrange = size(H,2)
    )
    synchronize(CUDABackend())
end

@kernel function ker_build_condensed_aug_symbolic_hess(sym, sym2, @Const(colptr), @Const(rowval))
    i = @index(Global)
    @inbounds for j in colptr[i]:colptr[i+1]-1
        c = rowval[j]
        sym[j] = (0,j,0)
        sym2[j] = (c,i)
    end
end

# function MadNLP._build_condensed_aug_symbolic_jt(Jt::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}, sym, sym2) where {Tv,Ti}
#     sym_cpu = Array(sym)
#     sym2_cpu = Array(sym2)
#     MadNLP._build_condensed_aug_symbolic_jt(
#         MadNLP.SparseMatrixCSC(Jt),
#         sym_cpu,
#         sym2_cpu
#     )

#     copyto!(sym, sym_cpu)
#     copyto!(sym2, sym2_cpu)
# end

@kernel function ker_build_condensed_aug_symbolic_jt(@Const(colptr), @Const(rowval), @Const(offsets), sym, sym2) 
    i = @index(Global)
    @inbounds cnt = if i==1
        0
    else
        offsets[i-1]
    end
    for j in colptr[i]:colptr[i+1]-1
        c1 = rowval[j]
        for k in j:colptr[i+1]-1
            c2 = rowval[k]
            cnt += 1
            sym[cnt] = (i,j,k)
            sym2[cnt] = (c2,c1)
        end
    end
end

function MadNLP._build_condensed_aug_symbolic_jt(Jt::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv,Ti}, sym, sym2) where {Tv,Ti}

    _offsets = map((i,j) -> div((j-i)^2 + (j-i), 2), @view(Jt.colPtr[1:end-1]) , @view(Jt.colPtr[2:end]))
    offsets = cumsum(_offsets)

    ker_build_condensed_aug_symbolic_jt(CUDABackend())(Jt.colPtr, Jt.rowVal, offsets, sym, sym2; ndrange = size(Jt,2))
    synchronize(CUDABackend())
end

function MadNLP._first_and_last_col(sym2::CuVector,ptr2)
    CUDA.@allowscalar begin
        first= sym2[1][2]
        last = sym2[ptr2[end]][2]
    end
    return (first, last)
end

MadNLP.nzval(H::CUDA.CUSPARSE.CuSparseMatrixCSC) = H.nzVal

function MadNLP._set_colptr!(colptr::CuVector, ptr2, sym2, guide)
    if length(ptr2) == 1 # otherwise error is thrown
        return
    end

    ker_set_colptr(CUDABackend())(
        colptr,
        sym2,
        ptr2,
        guide;
        ndrange = length(ptr2)-1
    )
    synchronize(CUDABackend())
    return
end


@kernel function ker_set_colptr(colptr, @Const(sym2), @Const(ptr2), @Const(guide))
    idx = @index(Global)
    @inbounds begin
        i = ptr2[idx+1]

        (~, prevcol) = sym2[i-1]
        (row, col)   = sym2[i]
        g = guide[i]
        for j in prevcol+1:col
            colptr[j] = g
        end
    end
end

function MadNLP._get_sparse_csc(dims, colptr::CuVector, rowval, nzval)
    return CUDA.CUSPARSE.CuSparseMatrixCSC(
        colptr,
        rowval,
        nzval,
        dims,
    )
end
function MadNLP.tril_to_full!(dense::CuMatrix{T}) where T
    n = size(dense,1)
    _tril_to_full!(CUDABackend())(dense; ndrange = div(n^2 + n,2))
end

@kernel function _tril_to_full!(dense)
    idx = @index(Global)
    n = size(dense,1)
    i,j = getij(idx,n)

    @inbounds dense[j,i] = dense[i,j]
end

function getij(idx,n)
    j = ceil(Int,((2n+1)-sqrt((2n+1)^2-8*idx))/2)
    i = idx-div((j-1)*(2n-j),2)
    return (i,j)
end



function MadNLP.force_lower_triangular!(I::CuVector{T},J) where T
    _force_lower_triangular!(CUDABackend())(I,J; ndrange=length(I))
end

@kernel function _force_lower_triangular!(I,J)
    i = @index(Global)

    @inbounds if J[i] > I[i]
        tmp=J[i]
        J[i]=I[i]
        I[i]=tmp
    end
end

if VERSION < v"1.10"
    function MadNLP.mul_hess_blk!(wx::CuVector{T}, kkt::Union{MadNLP.DenseKKTSystem,MadNLP.DenseCondensedKKTSystem}, t) where T
        n = size(kkt.hess, 1)
        CUDA.CUBLAS.symv!('L', one(T), kkt.hess, @view(t[1:n]), zero(T), @view(wx[1:n]))
        fill!(@view(wx[n+1:end]), 0)
        wx .+= t .* kkt.pr_diag
    end
end
