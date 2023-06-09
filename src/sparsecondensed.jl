"""
    SparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractCondensedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    pr_diag::VT
    du_diag::VT
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT}
    aug_com::MT
    aug_csc_map::Union{Nothing, Vector{Int}}
    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT}
    jac_com::MT
    jac_csc_map::Union{Nothing, Vector{Int}}
    # New
    hess_coo::SparseMatrixCOO{T,Int32,VT}
    hess_csc::MT
    hess_csc_map::Union{Nothing, Vector{Int}}
    jt_coo::SparseMatrixCOO{T,Int32,VT}
    jt_csc::MT
    jt_csc_map::Union{Nothing, Vector{Int}}
    aug_compressed::MT
    hptr::Vector{Tuple{Int,Int}}
    jptr::Vector{Tuple{Int,Tuple{Int,Int,Int}}}
    # Info
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    ind_aug_fixed::Vector{Int}
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

    aug_vec_length = n_tot+m
    aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

    I = Vector{Int32}(undef, aug_mat_length)
    J = Vector{Int32}(undef, aug_mat_length)
    V = VT(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    offset = n_tot+n_jac+n_slack+n_hess+m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    pr_diag = _madnlp_unsafe_wrap(V, n_tot)
    du_diag = _madnlp_unsafe_wrap(V, m, n_jac+n_slack+n_hess+n_tot+1)

    hess = _madnlp_unsafe_wrap(V, n_hess, n_tot+1)
    jac = _madnlp_unsafe_wrap(V, n_jac+n_slack, n_hess+n_tot+1)
    jac_callback = _madnlp_unsafe_wrap(V, n_jac, n_hess+n_tot+1)

    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )

    aug_com = MT(aug_raw)
    jac_com = MT(jac_raw)

    aug_csc_map = get_mapping(aug_com, aug_raw)
    jac_csc_map = get_mapping(jac_com, jac_raw)

    ind_aug_fixed = if isa(aug_com, SparseMatrixCSC)
        _get_fixed_variable_index(aug_com, ind_fixed)
    else
        zeros(Int, 0)
    end
    jac_scaling = ones(T, n_jac+n_slack)

    quasi_newton = QN(n)

    hess_coo = SparseMatrixCOO(n_tot, n_tot, hess_sparsity_I, hess_sparsity_J, hess)
    hess_csc = MT(hess_coo)
    hess_csc_map = get_mapping(hess_csc, hess_coo)

    jt_coo = SparseMatrixCOO(
        n_tot, m, 
        jac_sparsity_J,
        jac_sparsity_I,
        jac,
    )
    jt_csc = MT(jt_coo)
    jt_csc_map = get_mapping(jt_csc, jt_coo)
    
    aug_compressed, hptr, jptr = hpjtsj_symbolic(
        hess_csc,
        jt_csc
    )

    return SparseCondensedKKTSystem{T, VT, MT, QN}(
        hess, jac_callback, jac, quasi_newton, pr_diag, du_diag,
        aug_raw, aug_com, aug_csc_map,
        jac_raw, jac_com, jac_csc_map,
        hess_coo,hess_csc,hess_csc_map,
        jt_coo,jt_csc,jt_csc_map,
        aug_compressed, hptr, jptr,
        ind_ineq, ind_fixed, ind_aug_fixed, jac_scaling,
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

    cnt.linear_solver_time += @elapsed begin
        result = solve_refine!(x, solver.iterator, b)
    end

    if result == :Solved
        solve_status =  true
    else
        if improve!(solver.linear_solver)
            cnt.linear_solver_time += @elapsed begin
                factorize!(solver.linear_solver)
                ret = solve_refine!(x, solver.iterator, b)
                solve_status = (ret == :Solved)
            end
        else
            solve_status = false
        end
    end
    fixed_variable_treatment_vec!(full(x), solver.ind_fixed)
    return solve_status
end

function solve_refine!(
    x::AbstractKKTVector{T, VT},
    solver::RichardsonIterator{T, VT, KKT, LinSolver},
    b::AbstractKKTVector{T, VT},
) where {T, VT, KKT<:SparseCondensedKKTSystem, LinSolver}
    solve_refine!(primal_dual(x), solver, primal_dual(b))
end


nnz_jacobian(kkt::SparseCondensedKKTSystem) = nnz(kkt.jac_raw)
function compress_jacobian!(kkt::SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:SparseMatrixCSC{T, Int32}}
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    kkt.jac .*= kkt.jacobian_scaling # scaling
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
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
    mul!(y, Symmetric(kkt.aug_com, :L), x)
end
function mul!(y::AbstractKKTVector, kkt::SparseCondensedKKTSystem, x::AbstractKKTVector)
    mul!(full(y), Symmetric(kkt.aug_com, :L), full(x))
end
function jtprod!(y::AbstractVector, kkt::SparseCondensedKKTSystem, x::AbstractVector)
    mul!(y, kkt.jt_csc, x)
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

@inbounds function hpjtsj_symbolic(H::SparseMatrixCSC{Tv,Ti}, Jt::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti}
    nnzjtsj = _sym_length(Jt)
    
    sym = Vector{Tuple{Int,Int,Int}}(
        undef,
        nnz(H) + nnzjtsj
    )
    sym2 = Vector{Tuple{Int,Int}}(
        undef,
        nnz(H) + nnzjtsj
    )

    cnt = 0
    for i in 1:size(H,2)
        for j in H.colptr[i]:H.colptr[i+1]-1
            sym[cnt+=1] = (0,j,0)
            sym2[cnt] = (H.rowval[j],i)
        end
    end
# @inbounds function hpjtsj_symbolic(H::SparseMatrixCOO{Tv,Ti}, Jt::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti}
    
#     nnzjtsj = _sym_length(Jt)
    
#     sym = Vector{Tuple{Ti,Ti,Ti}}(
#         undef,
#         nnz(H) + nnzjtsj
#     )
#     sym2 = Vector{Tuple{Ti,Ti}}(
#         undef,
#         nnz(H) + nnzjtsj
#     )

#     cnt = 0
#     for (k,(i,j,v)) in enumerate(zip(H.I,H.J,H.V))
#         sym[cnt+=1] = (0,k,0)
#         sym2[cnt] = (i,j)
#     end

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

    hptr = Vector{Tuple{Ti,Ti}}(undef,nnz(H))
    jptr = Vector{Tuple{Ti,Tuple{Ti,Ti,Ti}}}(undef,nnzjtsj)

    colptr = ones(Ti,size(H,1)+1)
    rowval = Ti[]

    a = (0,0)
    cnt = 0
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

        if tuple[1] == 0
            hptr[hcnt += 1] = (cnt, tuple[2])
        else
            jptr[jcnt += 1] = (cnt, tuple)
        end
    end

    fill!(@view(colptr[prevcol+1:end]), cnt+1)

    hpjtsj = SparseMatrixCSC{Tv,Ti}(
        size(H)...,
        colptr, rowval, zeros(cnt)
    )

    return hpjtsj, hptr, jptr
end

@inbounds function hpjtsj_coord!(hpjtsj::SparseMatrixCSC{Tv,Ti}, H, Jt, S, hptr, jptr) where {Tv, Ti}
    fill!(hpjtsj.nzval, zero(Tv))
    
    @simd for idx in eachindex(hptr)
        i,j = hptr[idx]
        hpjtsj.nzval[i] = H.nzval[j]
    end
    
    @simd for idx in eachindex(jptr)
        (i,(j,k,l)) = jptr[idx]
        hpjtsj.nzval[i] += S[j] * Jt.nzval[k] * Jt.nzval[l]
    end
end

