#=
    GPU wrappers for SparseCondensedKKTSystem.
=#

function MadNLP.mul!(
    w::MadNLP.AbstractKKTVector{T,VT},
    kkt::MadNLP.SparseCondensedKKTSystem,
    x::MadNLP.AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T,VT<:CuVector{T}}
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

    MadNLP.mul!(wx, kkt.hess_com, xx, alpha, beta)
    MadNLP.mul!(wx, kkt.hess_com', xx, alpha, one(T))
    MadNLP.mul!(wx, kkt.jt_csc, xz, alpha, beta)
    if !isempty(kkt.ext.diag_map_to)
        _diag_operation_kernel!(CUDABackend())(
            wx,
            kkt.hess_com.nzVal,
            xx,
            alpha,
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to),
        )
        synchronize(CUDABackend())
    end

    MadNLP.mul!(wz, kkt.jt_csc', xx, alpha, one(T))
    MadNLP.axpy!(-alpha, xz, ws)
    MadNLP.axpy!(-alpha, xs, wz)
    return MadNLP._kktmul!(
        w,
        x,
        kkt.reg,
        kkt.du_diag,
        kkt.l_lower,
        kkt.u_lower,
        kkt.l_diag,
        kkt.u_diag,
        alpha,
        beta,
    )
end

function MadNLP.mul_hess_blk!(
    wx::VT,
    kkt::Union{MadNLP.SparseKKTSystem,MadNLP.SparseCondensedKKTSystem},
    t,
) where {T,VT<:CuVector{T}}
    n = size(kkt.hess_com, 1)
    wxx = @view(wx[1:n])
    tx = @view(t[1:n])

    MadNLP.mul!(wxx, kkt.hess_com, tx, one(T), zero(T))
    MadNLP.mul!(wxx, kkt.hess_com', tx, one(T), one(T))
    if !isempty(kkt.ext.diag_map_to)
        _diag_operation_kernel!(CUDABackend())(
            wxx,
            kkt.hess_com.nzVal,
            tx,
            one(T),
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to),
        )
        synchronize(CUDABackend())
    end

    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
    return
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

function MadNLP.get_sparse_condensed_ext(
    ::Type{VT},
    hess_com,
    jptr,
    jt_map,
    hess_map,
) where {T,VT<:CuVector{T}}
    hess_com_ptr = map((i, j) -> (i, j), hess_map, 1:length(hess_map))
    if length(hess_com_ptr) > 0 # otherwise error is thrown
        sort!(hess_com_ptr)
    end

    jt_csc_ptr = map((i, j) -> (i, j), jt_map, 1:length(jt_map))
    if length(jt_csc_ptr) > 0 # otherwise error is thrown
        sort!(jt_csc_ptr)
    end

    by = (i, j) -> i[1] != j[1]
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

function get_diagonal_mapping(colptr, rowval)
    nnz = length(rowval)
    if nnz == 0
        return similar(colptr, 0), similar(colptr, 0)
    end
    inds1 = findall(
        map(
            (x, y) -> ((x <= nnz) && (x != y)),
            @view(colptr[1:end-1]),
            @view(colptr[2:end])
        ),
    )
    if length(inds1) == 0
        return similar(rows, 0), similar(ptrs, 0)
    end
    ptrs = colptr[inds1]
    rows = rowval[ptrs]
    inds2 = findall(inds1 .== rows)
    if length(inds2) == 0
        return similar(rows, 0), similar(ptrs, 0)
    end

    return rows[inds2], ptrs[inds2]
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

function MadNLP._first_and_last_col(sym2::CuVector, ptr2)
    CUDA.@allowscalar begin
        first = sym2[1][2]
        last = sym2[ptr2[end]][2]
    end
    return (first, last)
end

MadNLP.nzval(H::CUSPARSE.CuSparseMatrixCSC) = H.nzVal

function MadNLP._get_sparse_csc(dims, colptr::CuVector, rowval, nzval)
    return CUSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function getij(idx, n)
    j = ceil(Int, ((2n + 1) - sqrt((2n + 1)^2 - 8 * idx)) / 2)
    i = idx - div((j - 1) * (2n - j), 2)
    return (i, j)
end



#=
    MadNLP._set_colptr!
=#

@kernel function _set_colptr_kernel!(colptr, @Const(sym2), @Const(ptr2), @Const(guide))
    idx = @index(Global)
    @inbounds begin
        i = ptr2[idx+1]

        (~, prevcol) = sym2[i-1]
        (row, col) = sym2[i]
        g = guide[i]
        for j in prevcol+1:col
            colptr[j] = g
        end
    end
end

function MadNLP._set_colptr!(colptr::CuVector, ptr2, sym2, guide)
    if length(ptr2) > 1 # otherwise error is thrown
        _set_colptr_kernel!(CUDABackend())(
            colptr,
            sym2,
            ptr2,
            guide;
            ndrange = length(ptr2) - 1,
        )
        synchronize(CUDABackend())
    end
    return
end


#=
    MadNLP.tril_to_full!
=#

@kernel function _tril_to_full_kernel!(dense)
    idx = @index(Global)
    n = size(dense, 1)
    i, j = getij(idx, n)
    @inbounds dense[j, i] = dense[i, j]
end

function MadNLP.tril_to_full!(dense::CuMatrix{T}) where {T}
    n = size(dense, 1)
    _tril_to_full_kernel!(CUDABackend())(dense; ndrange = div(n^2 + n, 2))
    synchronize(CUDABackend())
    return
end

#=
    MadNLP.force_lower_triangular!
=#

@kernel function _force_lower_triangular_kernel!(I, J)
    i = @index(Global)

    @inbounds if J[i] > I[i]
        tmp = J[i]
        J[i] = I[i]
        I[i] = tmp
    end
end

function MadNLP.force_lower_triangular!(I::CuVector{T}, J) where {T}
    if !isempty(I)
        _force_lower_triangular_kernel!(CUDABackend())(I, J; ndrange = length(I))
        synchronize(CUDABackend())
    end
    return
end

#=
    MadNLP.coo_to_csc
=#

@kernel function _set_coo_to_colptr_kernel!(colptr, @Const(coord))
    index = @index(Global)

    @inbounds begin
        if index == 1
            ((i2, j2), k2) = coord[index]
            for k in 1:j2
                colptr[k] = 1
            end
            if index == length(coord)
                ip1 = index + 1
                for k in j2+1:length(colptr)
                    colptr[k] = ip1
                end
            end
        else
            ((i1, j1), k1) = coord[index-1]
            ((i2, j2), k2) = coord[index]
            if j1 != j2
                for k in j1+1:j2
                    colptr[k] = index
                end
            end
            if index == length(coord)
                ip1 = index + 1
                for k in j2+1:length(colptr)
                    colptr[k] = ip1
                end
            end
        end
    end
end

@kernel function _set_coo_to_csc_map_kernel!(cscmap, @Const(mapptr), @Const(coord))
    index = @index(Global)
    @inbounds for l in mapptr[index]:mapptr[index+1]-1
        ((i, j), k) = coord[l]
        cscmap[k] = index
    end
end

function MadNLP.coo_to_csc(
    coo::MadNLP.SparseMatrixCOO{T,I,VT,VI},
) where {T,I,VT<:CuArray,VI<:CuArray}
    coord = map((i, j, k) -> ((i, j), k), coo.I, coo.J, 1:length(coo.I))
    if length(coord) > 0
        sort!(coord, lt = (((i, j), k), ((n, m), l)) -> (j, i) < (m, n))
    end

    mapptr = MadNLP.getptr(coord; by = ((x1, x2), (y1, y2)) -> x1 != y1)

    colptr = similar(coo.I, size(coo, 2) + 1)

    coord_csc = coord[@view(mapptr[1:end-1])]

    if length(coord_csc) > 0
        _set_coo_to_colptr_kernel!(CUDABackend())(
            colptr,
            coord_csc,
            ndrange = length(coord_csc),
        )
        synchronize(CUDABackend())
    else
        fill!(colptr, one(Int))
    end

    rowval = map(x -> x[1][1], coord_csc)
    nzval = similar(rowval, T)

    csc = CUSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, size(coo))

    cscmap = similar(coo.I, Int)
    if length(mapptr) > 1
        _set_coo_to_csc_map_kernel!(CUDABackend())(
            cscmap,
            mapptr,
            coord,
            ndrange = length(mapptr) - 1,
        )
        synchronize(CUDABackend())
    end

    return csc, cscmap
end

#=
    MadNLP.build_condensed_aug_coord!
=#

@kernel function _transfer_hessian_kernel!(y, @Const(ptr), @Const(x))
    index = @index(Global)
    @inbounds begin
        i, j = ptr[index]
        Atomix.@atomic y[i] += x[j]
    end
end

@kernel function _transfer_jtsj_kernel!(y, @Const(ptr), @Const(ptrptr), @Const(x), @Const(s))
    index = @index(Global)
    @inbounds for index2 in ptrptr[index]:ptrptr[index+1]-1
        i, (j, k, l) = ptr[index2]
        Atomix.@atomic y[i] += s[j] * x[k] * x[l]
    end
end

function MadNLP.build_condensed_aug_coord!(
    kkt::MadNLP.AbstractCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T}}
    fill!(kkt.aug_com.nzVal, zero(T))
    if length(kkt.hptr) > 0
        _transfer_hessian_kernel!(CUDABackend())(
            kkt.aug_com.nzVal,
            kkt.hptr,
            kkt.hess_com.nzVal;
            ndrange = length(kkt.hptr),
        )
        synchronize(CUDABackend())
    end
    if length(kkt.dptr) > 0
        _transfer_hessian_kernel!(CUDABackend())(
            kkt.aug_com.nzVal,
            kkt.dptr,
            kkt.pr_diag;
            ndrange = length(kkt.dptr),
        )
        synchronize(CUDABackend())
    end
    if length(kkt.ext.jptrptr) > 1 # otherwise error is thrown
        _transfer_jtsj_kernel!(CUDABackend())(
            kkt.aug_com.nzVal,
            kkt.jptr,
            kkt.ext.jptrptr,
            kkt.jt_csc.nzVal,
            kkt.diag_buffer;
            ndrange = length(kkt.ext.jptrptr) - 1,
        )
        synchronize(CUDABackend())
    end
    return
end

@kernel function _diag_operation_kernel!(
    y,
    @Const(A),
    @Const(x),
    @Const(alpha),
    @Const(idx_to),
    @Const(idx_fr)
)
    i = @index(Global)
    @inbounds begin
        to = idx_to[i]
        fr = idx_fr[i]
        Atomix.@atomic y[to] -= alpha * A[fr] * x[to]
    end
end

#=
    MadNLP.compress_hessian! / MadNLP.compress_jacobian!
=#

@kernel function _transfer_to_csc_kernel!(y, @Const(ptr), @Const(ptrptr), @Const(x))
    index = @index(Global)
    @inbounds for index2 in ptrptr[index]:ptrptr[index+1]-1
        i, j = ptr[index2]
        Atomix.@atomic y[i] += x[j]
    end
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.AbstractSparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    fill!(kkt.hess_com.nzVal, zero(T))
    if length(kkt.ext.hess_com_ptrptr) > 1
        _transfer_to_csc_kernel!(CUDABackend())(
            kkt.hess_com.nzVal,
            kkt.ext.hess_com_ptr,
            kkt.ext.hess_com_ptrptr,
            kkt.hess_raw.V;
            ndrange = length(kkt.ext.hess_com_ptrptr) - 1,
        )
        synchronize(CUDABackend())
    end
    return
end

function MadNLP.compress_jacobian!(
    kkt::MadNLP.SparseCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUDA.CUSOLVER.CuSparseMatrixCSC{T,Int32}}
    fill!(kkt.jt_csc.nzVal, zero(T))
    if length(kkt.ext.jt_csc_ptrptr) > 1 # otherwise error is thrown
        _transfer_to_csc_kernel!(CUDABackend())(
            kkt.jt_csc.nzVal,
            kkt.ext.jt_csc_ptr,
            kkt.ext.jt_csc_ptrptr,
            kkt.jt_coo.V;
            ndrange = length(kkt.ext.jt_csc_ptrptr) - 1,
        )
        synchronize(CUDABackend())
    end
    return
end

#=
    MadNLP._set_con_scale_sparse!
=#

@kernel function _set_con_scale_sparse_kernel!(
    con_scale,
    @Const(ptr),
    @Const(inds),
    @Const(jac_I),
    @Const(jac_buffer)
)
    index = @index(Global)

    @inbounds begin
        rng = ptr[index]:ptr[index+1]-1

        for k in rng
            (row, i) = inds[k]
            con_scale[row] = max(con_scale[row], abs(jac_buffer[i]))
        end
    end
end

function MadNLP._set_con_scale_sparse!(
    con_scale::VT,
    jac_I,
    jac_buffer,
) where {T,VT<:CuVector{T}}
    inds = map((i, j) -> (i, j), jac_I, 1:length(jac_I))
    if !isempty(inds)
        sort!(inds)
    end
    ptr = MadNLP.getptr(inds; by = ((x1, x2), (y1, y2)) -> x1 != y1)
    if length(ptr) > 1
        _set_con_scale_sparse_kernel!(CUDABackend())(
            con_scale,
            ptr,
            inds,
            jac_I,
            jac_buffer;
            ndrange = length(ptr) - 1,
        )
        synchronize(CUDABackend())
    end
    return
end

#=
    MadNLP._build_condensed_aug_symbolic_hess
=#

@kernel function _build_condensed_aug_symbolic_hess_kernel!(
    sym,
    sym2,
    @Const(colptr),
    @Const(rowval)
)
    i = @index(Global)
    @inbounds for j in colptr[i]:colptr[i+1]-1
        c = rowval[j]
        sym[j] = (0, j, 0)
        sym2[j] = (c, i)
    end
end

function MadNLP._build_condensed_aug_symbolic_hess(
    H::CUSPARSE.CuSparseMatrixCSC{Tv,Ti},
    sym,
    sym2,
) where {Tv,Ti}
    if size(H, 2) > 0
        _build_condensed_aug_symbolic_hess_kernel!(CUDABackend())(
            sym,
            sym2,
            H.colPtr,
            H.rowVal;
            ndrange = size(H, 2),
        )
        synchronize(CUDABackend())
    end
    return
end

#=
    MadNLP._build_condensed_aug_symbolic_jt
=#

@kernel function _build_condensed_aug_symbolic_jt_kernel!(
    sym,
    sym2,
    @Const(colptr),
    @Const(rowval),
    @Const(offsets)
)
    i = @index(Global)
    @inbounds begin
        cnt = if i == 1
            0
        else
            offsets[i-1]
        end
        for j in colptr[i]:colptr[i+1]-1
            c1 = rowval[j]
            for k in j:colptr[i+1]-1
                c2 = rowval[k]
                cnt += 1
                sym[cnt] = (i, j, k)
                sym2[cnt] = (c2, c1)
            end
        end
    end
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
        _build_condensed_aug_symbolic_jt_kernel!(CUDABackend())(
            sym,
            sym2,
            Jt.colPtr,
            Jt.rowVal,
            offsets;
            ndrange = size(Jt, 2),
        )
        synchronize(CUDABackend())
    end
    return
end

