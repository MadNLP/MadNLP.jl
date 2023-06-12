"""
    SparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractCondensedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}
    # Hessian
    hess::VT
    hess_coo::SparseMatrixCOO{T,Int32,VT}
    hess_csc::MT
    hess_csc_map::Union{Nothing, Vector{Int}}

    # Jacobian
    jac::VT
    jac_raw::SparseMatrixCOO{T,Int32,VT}
    jt_coo::SparseMatrixCOO{T,Int32,VT}
    jt_csc::MT
    jt_csc_map::Union{Nothing, Vector{Int}}
    
    quasi_newton::QN
    pr_diag::VT
    du_diag::VT
    # Augmented system
    aug_com::MT
    
    # slack diagonal buffer
    diag_buffer::VT
    # aug_compressed::MT
    dptr::Vector{Tuple{Int,Int}}
    hptr::Vector{Tuple{Int,Int}}
    jptr::Vector{Tuple{Int,Tuple{Int,Int,Int}}}
    # Info
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    jacobian_scaling::VT
end

#=
    SparseCondensedKKTSystem
=#

function SparseCondensedKKTSystem{T, VT, MT, QN}(
    n::Int, m::Int, ind_ineq::Vector{Int}, ind_fixed::Vector{Int},
    hess_sparsity_I, hess_sparsity_J,
    jac_sparsity_I, jac_sparsity_J,
) where {T, VT, MT, QN}
    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    pr_diag = zeros(n_tot)
    du_diag = zeros(m)

    hess = zeros(n_hess)
    jac = zeros(n_jac+n_slack)

    jac_raw = SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )

    jac_scaling = ones(T, n_jac+n_slack)

    quasi_newton = QN(n)

    diag_buffer = VT(undef,m)
    hess_coo = SparseMatrixCOO(n, n, hess_sparsity_I, hess_sparsity_J, hess)
    hess_csc = MT(hess_coo)
    hess_csc_map = get_mapping(hess_csc, hess_coo)

    jt_coo = SparseMatrixCOO(
        n, m, 
        jac_sparsity_J,
        jac_sparsity_I,
        jac,
    )
    jt_csc = MT(jt_coo)
    jt_csc_map = get_mapping(jt_csc, jt_coo)
    
    aug_com, dptr, hptr, jptr = aug_com_symbolic(
        hess_csc,
        jt_csc
    )

    return SparseCondensedKKTSystem{T, VT, MT, QN}(
        hess, hess_coo,hess_csc,hess_csc_map,
        jac, jac_raw, jt_coo,jt_csc,jt_csc_map,
        quasi_newton, pr_diag, du_diag,
        aug_com, diag_buffer, dptr, hptr, jptr,
        ind_ineq, ind_fixed, jac_scaling,
    )
end

# Build KKT system directly from AbstractNLPModel
function SparseCondensedKKTSystem{T, VT, MT, QN}(nlp::AbstractNLPModel, ind_cons=get_index_constraints(nlp)) where {T, VT, MT, QN}
    n_slack = length(ind_cons.ind_ineq)
    # Deduce KKT size.

    n = get_nvar(nlp)
    m = get_ncon(nlp)
    # Evaluate sparsity pattern
    jac_I = Vector{Int32}(undef, get_nnzj(nlp.meta))
    jac_J = Vector{Int32}(undef, get_nnzj(nlp.meta))
    jac_structure!(nlp,jac_I, jac_J)

    hess_I, hess_J = build_hessian_structure(nlp, QN)

    force_lower_triangular!(hess_I,hess_J)
    return SparseCondensedKKTSystem{T, VT, MT, QN}(
        n, m, ind_cons.ind_ineq, ind_cons.ind_fixed,
        hess_I, hess_J, jac_I, jac_J,
    )
end

is_reduced(::SparseCondensedKKTSystem) = true
num_variables(kkt::SparseCondensedKKTSystem) = length(kkt.pr_diag)


function solve_refine_wrapper!(
    solver::MadNLPSolver{T,<:SparseCondensedKKTSystem},
    x::AbstractKKTVector,
    b::AbstractKKTVector,
) where T
    cnt = solver.cnt
    @trace(solver.logger,"Iterative solution started.")
    fixed_variable_treatment_vec!(full(b), solver.ind_fixed)

    kkt = solver.kkt
    n = size(kkt.hess_csc, 1)
    m = size(kkt.jt_csc, 2)

    # load buffers
    b_c = view(full(solver._w1), 1:n)
    x_c = view(full(solver._w2), 1:n)
    jv_x = view(full(solver._w3), 1:m) 
    jv_t = view(primal(solver._w4), 1:n)
    v_c = dual(solver._w4)

    Σs = view(kkt.pr_diag, n+1:n+m)

    # Decompose right hand side
    bx = view(full(b), 1:n)
    bs = view(full(b), n+1:n+m)
    bz = view(full(b), n+m+1:n+2*m)

    # Decompose results
    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+m)
    xz = view(full(x), n+m+1:n+2*m)

    v_c .= (Σs .* bz .+ bs) 
    mul!(jv_t, kkt.jt_csc, v_c)

    # init right-hand-side
    b_c .= bx .+ jv_t

    cnt.linear_solver_time += @elapsed (result = solve_refine!(x_c, solver.iterator, b_c))

    # Expand solution
    xx .= x_c
    xs .= .-bz .+ mul!(jv_x, kkt.jt_csc', xx)
    xz .= .-bs .+ Σs .* xs

    fixed_variable_treatment_vec!(full(x), solver.ind_fixed)
    return (result == :Solved)
end

function is_inertia_correct(kkt::SparseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0)
end
Base.size(kkt::SparseCondensedKKTSystem,n::Int) = size(kkt.aug_com,n)


nnz_jacobian(kkt::SparseCondensedKKTSystem) = nnz(kkt.jac_raw)
function compress_jacobian!(kkt::SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:SparseMatrixCSC{T, Int32}}
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    kkt.jac .*= kkt.jacobian_scaling # scaling
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function set_jacobian_scaling!(kkt::SparseCondensedKKTSystem{T, VT, MT}, constraint_scaling::AbstractVector) where {T, VT, MT}
    nnzJ = length(kkt.jac)::Int
    @inbounds for i in 1:nnzJ
        index = kkt.jac_raw.I[i]
        kkt.jacobian_scaling[i] = constraint_scaling[index]
    end
end


function mul!(y::AbstractVector, kkt::SparseCondensedKKTSystem, x::AbstractVector)
    # TODO: implement properly with AbstractKKTRHS
    if length(y) == length(x) == size(kkt.aug_com, 1)
        mul!(y, Symmetric(kkt.aug_com, :L), x)
        return
    end

    n = size(kkt.hess_csc, 1)
    m = size(kkt.jt_csc, 2)


    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, n+1:n+m)
    Σd = kkt.du_diag

    # Decompose x
    xx = view(x, 1:n)
    xs = view(x, 1+n:n+m)
    xy = view(x, 1+n+m:n+2*m)

    # Decompose y
    yx = view(y, 1:n)
    ys = view(y, 1+n:n+m)
    yy = view(y, 1+n+m:n+2*m)

    # / x (variable)
    mul!(yx, Symmetric(kkt.hess_csc, :L), xx)
    yx .+= Σx .* xx
    mul!(yx, kkt.jt_csc, xy, 1.0, 1.0)

    # / s (slack)
    ys .= Σs .* xs
    ys .-= xy

    # / y (multiplier)
    yy .= Σd .* xy
    mul!(yy, kkt.jt_csc', xx, 1.0, 1.0)
    yy .-= xs

end

function jtprod!(y::AbstractVector, kkt::SparseCondensedKKTSystem, x::AbstractVector)
    n = size(kkt.hess_csc, 1)
    m = size(kkt.jt_csc, 2)

    mul!(view(y, 1:n), kkt.jt_csc, x)
    y[size(kkt.jt_csc,1)+1:end] .= -x
end

function _sym_length(Jt)
    len = 0
    for i=1:size(Jt,2)
        n = Jt.colptr[i+1] - Jt.colptr[i]
        len += div(n^2 + n, 2)
    end
    return len
end

@inbounds function aug_com_symbolic(H::SparseMatrixCSC{Tv,Ti}, Jt::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti}
    nnzjtsj = _sym_length(Jt)
    
    sym = Vector{Tuple{Int,Int,Int}}(
        undef,
        size(H,2) + nnz(H) + nnzjtsj
    )
    sym2 = Vector{Tuple{Int,Int}}(
        undef,
        size(H,2) + nnz(H) + nnzjtsj
    )

    cnt = 0
    for i in 1:size(H,2)
        sym[cnt+=1] = (-1,i,0)
        sym2[cnt] = (i,i)
    end
    
    for i in 1:size(H,2)
        for j in H.colptr[i]:H.colptr[i+1]-1
            sym[cnt+=1] = (0,j,0)
            sym2[cnt] = (H.rowval[j],i)
        end
    end

    for i in 1:size(Jt,2)
        for j in Jt.colptr[i]:Jt.colptr[i+1]-1
            for k in j:Jt.colptr[i+1]-1
                c1 = Jt.rowval[j]
                c2 = Jt.rowval[k]
                if c1 >= c2
                    sym[cnt+=1] = (i,j,k)
                    sym2[cnt] = (c1,c2)
                else
                    sym[cnt+=1] = (i,j,k)
                    sym2[cnt] = (c2,c1)
                end
            end
        end
    end
    p = sortperm(sym2; by = ((i,j),) -> (j,i), alg=Base.Sort.MergeSort)
    permute!(sym, p)
    permute!(sym2, p)

    dptr = Vector{Tuple{Ti,Ti}}(undef,size(H,2))
    hptr = Vector{Tuple{Ti,Ti}}(undef,nnz(H))
    jptr = Vector{Tuple{Ti,Tuple{Ti,Ti,Ti}}}(undef,nnzjtsj)

    colptr = ones(Ti,size(H,1)+1)
    rowval = Ti[]

    a = (0,0)
    cnt = 0
    dcnt = 0
    hcnt = 0
    jcnt = 0
    prevcol = 0
    
    for (new, tuple) in zip(sym2,sym)

        if new != a
            cnt += 1
            
            (row,col) = new
            push!(rowval, row)
            a = new
            if prevcol != col
                fill!(@view(colptr[prevcol+1:col]), cnt)
                prevcol = col
            end
        end

        if tuple[1] == -1
            dptr[dcnt += 1] = (cnt, tuple[2])
        elseif tuple[1] == 0
            hptr[hcnt += 1] = (cnt, tuple[2])
        else
            jptr[jcnt += 1] = (cnt, tuple)
        end
    end

    fill!(@view(colptr[prevcol+1:end]), cnt+1)

    aug_com = SparseMatrixCSC{Tv,Ti}(
        size(H)...,
        colptr, rowval, zeros(cnt)
    )

    return aug_com, dptr, hptr, jptr
end

@inbounds function aug_com_coord!(aug_com::SparseMatrixCSC{Tv,Ti}, pr_diag, H, Jt, diag_buffer, dptr, hptr, jptr) where {Tv, Ti}
    fill!(aug_com.nzval, zero(Tv))
    
    @simd for idx in eachindex(hptr)
        i,j = hptr[idx]
        aug_com.nzval[i] = H.nzval[j]
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


function build_kkt!(kkt::SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:SparseMatrixCSC{T, Int32}}
    transfer!(kkt.hess_csc, kkt.hess_coo, kkt.hess_csc_map)

    n = size(kkt.hess_csc, 1)
    m = size(kkt.jt_csc, 2)


    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, n+1:n+m)
    Σd = kkt.du_diag

    kkt.diag_buffer .= 1 ./ ( 1 ./ Σs .- Σd )
    aug_com_coord!(kkt.aug_com, kkt.pr_diag, kkt.hess_csc, kkt.jt_csc, kkt.S, kkt.dptr, kkt.hptr, kkt.jptr)

    treat_fixed_variable!(kkt)
end
