########################################################
##### oneAPI wrappers for SparseCondensedKKTSystem #####
########################################################

#=
    SparseMatrixCOO to oneMKLMatrixCSC
=#

function MadNLP.transfer!(
    dest::oneMKL.oneMKLMatrixCSC,
    src::MadNLP.SparseMatrixCOO,
    map,
)
    return copyto!(view(dest.nzVal, map), src.V)
end

#=
    MadNLP.mul! with SparseCondensedKKTSystem + oneAPI
=#

function MadNLP.mul!(
    w::MadNLP.AbstractKKTVector{T,VT},
    kkt::MadNLP.SparseCondensedKKTSystem,
    x::MadNLP.AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T,VT<:oneVector{T}}
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
        backend = OneAPIBackend()
        MadNLPGPU._diag_operation_kernel!(backend)(
            wx,
            kkt.hess_com.nzVal,
            xx,
            alpha,
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to),
        )
        synchronize(backend)
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
) where {T,VT<:oneVector{T}}
    n = size(kkt.hess_com, 1)
    wxx = @view(wx[1:n])
    tx = @view(t[1:n])

    MadNLP.mul!(wxx, kkt.hess_com, tx, one(T), zero(T))
    MadNLP.mul!(wxx, kkt.hess_com', tx, one(T), one(T))
    if !isempty(kkt.ext.diag_map_to)
        backend = OneAPIBackend()
        MadNLPGPU._diag_operation_kernel!(backend)(
            wxx,
            kkt.hess_com.nzVal,
            tx,
            one(T),
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to),
        )
        synchronize(backend)
    end

    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
    return
end

function MadNLP.get_tril_to_full(csc::oneMKL.oneMKLMatrixCSC{Tv,Ti}) where {Tv,Ti}
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
    return oneMKL.oneMKLMatrixCSC{Tv,Ti}(
        oneArray(cscind.colptr),
        oneArray(cscind.rowval),
        oneVector{Tv}(undef, MadNLP.nnz(cscind)),
        size(csc),
    ),
    view(csc.nzVal, oneArray(cscind.nzval))
end

function MadNLP.get_sparse_condensed_ext(
    ::Type{VT},
    hess_com,
    jptr,
    jt_map,
    hess_map,
) where {T,VT<:oneVector{T}}
    zvals = oneVector{Int}(1:length(hess_map))
    hess_com_ptr = map((i, j) -> (i, j), hess_map, zvals)
    if length(hess_com_ptr) > 0 # otherwise error is thrown
        sort!(hess_com_ptr)
    end

    jvals = oneVector{Int}(1:length(jt_map))
    jt_csc_ptr = map((i, j) -> (i, j), jt_map, jvals)
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

function MadNLP._sym_length(Jt::oneMKL.oneMKLMatrixCSC)
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

function MadNLP._first_and_last_col(sym2::oneVector, ptr2)
    AMDGPU.@allowscalar begin
        first = sym2[1][2]
        last = sym2[ptr2[end]][2]
    end
    return (first, last)
end

MadNLP.nzval(H::oneMKL.oneMKLMatrixCSC) = H.nzVal

function MadNLP._get_sparse_csc(dims, colptr::oneVector, rowval, nzval)
    return oneMKL.oneMKLMatrixCSC(colptr, rowval, nzval, dims)
end

function getij(idx, n)
    j = ceil(Int, ((2n + 1) - sqrt((2n + 1)^2 - 8 * idx)) / 2)
    i = idx - div((j - 1) * (2n - j), 2)
    return (i, j)
end

#=
    MadNLP._set_colptr!
=#

function MadNLP._set_colptr!(colptr::oneVector, ptr2, sym2, guide)
    if length(ptr2) > 1 # otherwise error is thrown
        backend = OneAPIBackend()
        MadNLPGPU._set_colptr_kernel!(backend)(
            colptr,
            sym2,
            ptr2,
            guide;
            ndrange = length(ptr2) - 1,
        )
        synchronize(backend)
    end
    return
end


#=
    MadNLP.tril_to_full!
=#

function MadNLP.tril_to_full!(dense::oneMatrix{T}) where {T}
    n = size(dense, 1)
    backend = OneAPIBackend()
    MadNLPGPU._tril_to_full_kernel!(backend)(dense; ndrange = div(n^2 + n, 2))
    synchronize(backend)
    return
end

#=
    MadNLP.force_lower_triangular!
=#

function MadNLP.force_lower_triangular!(I::oneVector{T}, J) where {T}
    if !isempty(I)
        backend = OneAPIBackend()
        MadNLPGPU._force_lower_triangular_kernel!(backend)(I, J; ndrange = length(I))
        synchronize(backend)
    end
    return
end

#=
    MadNLP.coo_to_csc
=#

function MadNLP.coo_to_csc(
    coo::MadNLP.SparseMatrixCOO{T,I,VT,VI},
) where {T,I,VT<:oneArray,VI<:oneArray}
    zvals = oneVector{Int}(1:length(coo.I))
    coord = map((i, j, k) -> ((i, j), k), coo.I, coo.J, zvals)
    if length(coord) > 0
        sort!(coord, lt = (((i, j), k), ((n, m), l)) -> (j, i) < (m, n))
    end

    mapptr = MadNLP.getptr(coord; by = ((x1, x2), (y1, y2)) -> x1 != y1)

    colptr = similar(coo.I, size(coo, 2) + 1)

    coord_csc = coord[@view(mapptr[1:end-1])]

    backend = OneAPIBackend()
    if length(coord_csc) > 0
        MadNLPGPU._set_coo_to_colptr_kernel!(backend)(
            colptr,
            coord_csc,
            ndrange = length(coord_csc),
        )
        synchronize(backend)
    else
        fill!(colptr, one(Int))
    end

    rowval = map(x -> x[1][1], coord_csc)
    nzval = similar(rowval, T)

    csc = oneMKL.oneMKLMatrixCSC(colptr, rowval, nzval, size(coo))

    cscmap = similar(coo.I, Int)
    if length(mapptr) > 1
        MadNLPGPU._set_coo_to_csc_map_kernel!(backend)(
            cscmap,
            mapptr,
            coord,
            ndrange = length(mapptr) - 1,
        )
        synchronize(backend)
    end

    return csc, cscmap
end

#=
    MadNLP.build_condensed_aug_coord!
=#

function MadNLP.build_condensed_aug_coord!(
    kkt::MadNLP.AbstractCondensedKKTSystem{T,VT,MT},
) where {T,VT,MT<:oneMKL.oneMKLMatrixCSC{T}}
    fill!(kkt.aug_com.nzVal, zero(T))
    backend = OneAPIBackend()
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

#=
    MadNLP.compress_hessian! / MadNLP.compress_jacobian!
=#

function MadNLP.compress_hessian!(
    kkt::MadNLP.AbstractSparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:oneMKL.oneMKLMatrixCSC{T,Int32}}
    fill!(kkt.hess_com.nzVal, zero(T))
    backend = OneAPIBackend()
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
) where {T,VT,MT<:oneMKL.oneMKLMatrixCSC{T,Int32}}
    fill!(kkt.jt_csc.nzVal, zero(T))
    backend = OneAPIBackend()
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

#=
    MadNLP._set_con_scale_sparse!
=#

function MadNLP._set_con_scale_sparse!(
    con_scale::VT,
    jac_I,
    jac_buffer,
) where {T,VT<:oneVector{T}}
    ind_jac = oneVector{Int}(1:length(jac_I))
    inds = map((i, j) -> (i, j), jac_I, ind_jac)
    !isempty(inds) && sort!(inds)
    ptr = MadNLP.getptr(inds; by = ((x1, x2), (y1, y2)) -> x1 != y1)
    if length(ptr) > 1
        backend = OneAPIBackend()
        MadNLPGPU._set_con_scale_sparse_kernel!(backend)(
            con_scale,
            ptr,
            inds,
            jac_I,
            jac_buffer;
            ndrange = length(ptr) - 1,
        )
        synchronize(backend)
    end
    return
end

#=
    MadNLP._build_condensed_aug_symbolic_hess
=#

function MadNLP._build_condensed_aug_symbolic_hess(
    H::oneMKL.oneMKLMatrixCSC{Tv,Ti},
    sym,
    sym2,
) where {Tv,Ti}
    if size(H, 2) > 0
        backend = OneAPIBackend()
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

#=
    MadNLP._build_condensed_aug_symbolic_jt
=#

function MadNLP._build_condensed_aug_symbolic_jt(
    Jt::oneMKL.oneMKLMatrixCSC{Tv,Ti},
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
        backend = OneAPIBackend()
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

#=
    MadNLP._build_scale_augmented_system_coo!
=#

function MadNLP._build_scale_augmented_system_coo!(dest, src, scaling::oneArray, n, m)
    backend = OneAPIBackend()
    MadNLPGPU._scale_augmented_system_coo_kernel!(backend)(
        dest.V,
        src.I,
        src.J,
        src.V,
        scaling,
        n,
        m;
        ndrange = nnz(src),
    )
    synchronize(backend)
    return
end
