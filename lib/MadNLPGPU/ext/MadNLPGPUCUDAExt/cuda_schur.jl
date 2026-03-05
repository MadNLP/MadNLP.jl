###########################################################
##### CUDA wrappers for SchurComplementKKTSystem ##########
###########################################################

"""
    GPUSchurComplementKKTSystem

GPU-native Schur complement KKT system. Uses a single batched CuSparseMatrixCSC
for all scenario blocks (factored by CUDSSSolver with uniform batching) and
CUBLAS strided batched GEMM for the Schur complement accumulation.
"""
struct GPUSchurComplementKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,   # linear solver for Schur complement S (LapackCUDASolver)
    LS2,  # batched scenario solver (CUDSSSolver)
    COO_T,   # SparseMatrixCOO type for hess_raw
    COO_JT,  # SparseMatrixCOO type for jt_coo
    CSC_T,   # CuSparseMatrixCSC type
    VI <: AbstractVector{Int},
} <: MadNLP.AbstractCondensedKKTSystem{T, VT, MT, QN}

    # COO value buffers on GPU
    hess::VT
    jac::VT

    # COO structures
    hess_raw::COO_T
    jt_coo::COO_JT

    # CSC representations on GPU
    hess_csc::CSC_T
    hess_csc_map::VI
    jt_csc::CSC_T
    jt_csc_map::VI

    quasi_newton::QN

    # Standard diagonals on GPU
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # Two-stage dimensions
    ns::Int
    nv::Int
    nd::Int
    nc::Int
    nc_eq_per_s::Int
    nc_ineq_per_s::Int
    blk_size::Int

    # Batched scenario block
    A_kk_batched::CSC_T
    nnz_per_scenario::Int

    # Dense blocks
    C_dk_batched::CuArray{T, 3}     # (nd, blk_size, ns)
    aug_com::MT                     # (nd, nd) — Schur complement S

    # Buffers
    diag_buffer::VT
    buffer::VT
    rhs_d::VT
    rhs_k_batched::MT                   # (blk_size, ns)
    tmp_blk_nd_batched::CuArray{T, 3}   # (blk_size, nd, ns)
    S_contrib::CuArray{T, 3}            # (nd, nd, ns)
    solve_buffer::VT                    # blk_size * ns

    # Flattened GPU index maps (all scenarios concatenated)
    # n_per_s_* fields store entries-per-scenario for kernel dispatch
    n_per_s_hess_Akk::Int
    gpu_hess_Akk_coo::VI
    gpu_hess_Akk_nzpos::VI

    n_per_s_hess_Cdk::Int
    gpu_hess_Cdk_coo::VI
    gpu_hess_Cdk_row::VI
    gpu_hess_Cdk_col::VI

    n_per_s_pr_diag::Int
    gpu_pr_diag_global::VI
    gpu_pr_diag_nzpos::VI

    n_per_s_du_diag::Int
    gpu_du_diag_global::VI
    gpu_du_diag_nzpos::VI

    n_per_s_jeq_Akk::Int
    gpu_jeq_Akk_coo::VI
    gpu_jeq_Akk_nzpos::VI

    n_per_s_jeq_Cdk::Int
    gpu_jeq_Cdk_coo::VI
    gpu_jeq_Cdk_row::VI
    gpu_jeq_Cdk_col::VI

    n_per_s_ineq_Akk::Int
    gpu_ineq_Akk_nzpos::VI
    gpu_ineq_Akk_jcoo1::VI
    gpu_ineq_Akk_jcoo2::VI
    gpu_ineq_Akk_bufidx::VI

    n_per_s_ineq_Cdk::Int
    gpu_ineq_Cdk_row::VI
    gpu_ineq_Cdk_col::VI
    gpu_ineq_Cdk_jcoo_d::VI
    gpu_ineq_Cdk_jcoo_v::VI
    gpu_ineq_Cdk_bufidx::VI

    n_per_s_ineq_S::Int
    gpu_ineq_S_row::VI
    gpu_ineq_S_col::VI
    gpu_ineq_S_jcoo1::VI
    gpu_ineq_S_jcoo2::VI
    gpu_ineq_S_bufidx::VI

    hess_S_coo::VI
    hess_S_row::VI
    hess_S_col::VI

    eq_global_indices::VI

    # Inequality/equality/bound indices
    n_eq::Int
    ind_eq::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # Solvers
    scenario_solver::LS2
    linear_solver::LS
    etc::Dict{Symbol, Any}
end

# --- Dispatch: GPU path when callback uses CuVector ---
function MadNLP.create_kkt_system(
    ::Type{MadNLP.SchurComplementKKTSystem},
    cb::MadNLP.SparseCallback{T, VT},
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
    schur_ns::Int=0,
    schur_nv::Int=0,
    schur_nd::Int=0,
    schur_nc::Int=0,
    schur_scenario_linear_solver::Type=CUDSSSolver,
) where {T, VT <: CuVector{T}}

    n = cb.nvar
    m = cb.ncon
    ns_ineq = length(cb.ind_ineq)
    n_eq_total = m - ns_ineq
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    @assert schur_ns > 0 "schur_ns must be specified and positive"
    @assert schur_nv > 0 "schur_nv must be specified and positive"
    @assert schur_nd > 0 "schur_nd must be specified and positive"
    @assert n == schur_ns * schur_nv + schur_nd
    @assert m == schur_ns * schur_nc

    ns = schur_ns
    nv = schur_nv
    nd = schur_nd
    nc = schur_nc

    # --- Get sparsity patterns on CPU ---
    jac_sparsity_I = Vector{Int32}(undef, cb.nnzj)
    jac_sparsity_J = Vector{Int32}(undef, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I_gpu, hess_sparsity_J_gpu = MadNLP.build_hessian_structure(cb, hessian_approximation)
    # Convert to CPU for construction-time classification loops
    hess_sparsity_I = Vector{Int32}(Array(hess_sparsity_I_gpu))
    hess_sparsity_J = Vector{Int32}(Array(hess_sparsity_J_gpu))
    MadNLP.force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    n_hess = length(hess_sparsity_I)
    n_jac = length(jac_sparsity_I)

    # --- COO value buffers on GPU ---
    hess = CuVector{T}(undef, n_hess)
    jac = CuVector{T}(undef, n_jac)
    fill!(hess, zero(T))
    fill!(jac, zero(T))

    # --- Build global COO + CSC on GPU ---
    hess_I_gpu = CuVector{Int32}(hess_sparsity_I)
    hess_J_gpu = CuVector{Int32}(hess_sparsity_J)
    jac_I_gpu = CuVector{Int32}(jac_sparsity_I)
    jac_J_gpu = CuVector{Int32}(jac_sparsity_J)

    hess_raw = MadNLP.SparseMatrixCOO(n, n, hess_I_gpu, hess_J_gpu, hess)
    jt_coo = MadNLP.SparseMatrixCOO(n, m, jac_J_gpu, jac_I_gpu, jac)

    hess_csc, hess_csc_map = MadNLP.coo_to_csc(hess_raw)
    jt_csc, jt_csc_map = MadNLP.coo_to_csc(jt_coo)

    # --- Classify constraints per scenario (on CPU) ---
    # Transfer index arrays to CPU for construction-time classification
    cpu_ind_eq = Array(cb.ind_eq)
    cpu_ind_ineq = Array(cb.ind_ineq)
    cpu_ind_lb = Array(cb.ind_lb)
    cpu_ind_ub = Array(cb.ind_ub)
    ind_eq_set = Set(cpu_ind_eq)
    ind_ineq_set = Set(cpu_ind_ineq)

    eq_per_scenario = Vector{Vector{Int}}(undef, ns)
    ineq_per_scenario = Vector{Vector{Int}}(undef, ns)

    nc_eq_per_s = 0
    nc_ineq_per_s = 0
    for k in 1:ns
        cr = (k-1)*nc+1 : k*nc
        eq_per_scenario[k] = Int[]
        ineq_per_scenario[k] = Int[]
        for gi in cr
            gi in ind_eq_set && push!(eq_per_scenario[k], gi)
            gi in ind_ineq_set && push!(ineq_per_scenario[k], gi)
        end
        if k == 1
            nc_eq_per_s = length(eq_per_scenario[k])
            nc_ineq_per_s = length(ineq_per_scenario[k])
        end
    end

    blk_size = nv + nc_eq_per_s

    # --- Build ineq lookup ---
    ineq_to_bufidx = Dict{Int,Int}()
    for idx in 1:length(cpu_ind_ineq)
        ineq_to_bufidx[cpu_ind_ineq[idx]] = idx
    end

    # --- Build Jacobian COO lookup ---
    jac_by_constraint = Dict{Int, Vector{Tuple{Int,Int}}}()
    for ci in 1:n_jac
        row = Int(jac_sparsity_I[ci])
        col = Int(jac_sparsity_J[ci])
        entries = get!(Vector{Tuple{Int,Int}}, jac_by_constraint, row)
        push!(entries, (ci, col))
    end

    d_start = ns * nv + 1
    d_end = ns * nv + nd

    # --- Classify Hessian COO entries (CPU) ---
    hess_S_coo_list = Int[]
    hess_S_row_list = Int[]
    hess_S_col_list = Int[]

    hess_per_scenario_diag = [Tuple{Int,Int,Int}[] for _ in 1:ns]
    hess_per_scenario_coupling = [Tuple{Int,Int,Int}[] for _ in 1:ns]

    for ci in 1:n_hess
        ri = Int(hess_sparsity_I[ci])
        rj = Int(hess_sparsity_J[ci])

        if ri >= d_start && ri <= d_end && rj >= d_start && rj <= d_end
            di = ri - d_start + 1
            dj = rj - d_start + 1
            push!(hess_S_coo_list, ci)
            push!(hess_S_row_list, di)
            push!(hess_S_col_list, dj)
            if di != dj
                push!(hess_S_coo_list, ci)
                push!(hess_S_row_list, dj)
                push!(hess_S_col_list, di)
            end
            continue
        end

        if ri >= d_start && ri <= d_end && rj < d_start
            di = ri - d_start + 1
            k2 = div(rj - 1, nv) + 1
            if k2 >= 1 && k2 <= ns
                vj = rj - (k2-1)*nv
                push!(hess_per_scenario_coupling[k2], (ci, di, vj))
            end
            continue
        end

        if ri < d_start && rj < d_start
            ki = div(ri - 1, nv) + 1
            kj = div(rj - 1, nv) + 1
            if ki == kj && ki >= 1 && ki <= ns
                li = ri - (ki-1)*nv
                lj = rj - (ki-1)*nv
                push!(hess_per_scenario_diag[ki], (ci, li, lj))
            end
        end
    end

    # --- Build scenario-1 A_kk sparsity (shared structure for all scenarios) ---
    vr_start_1 = 1
    eq_cons_1 = eq_per_scenario[1]
    ineq_cons_1 = ineq_per_scenario[1]

    eq_local_1 = Dict{Int,Int}()
    for (ci, gi) in enumerate(eq_cons_1)
        eq_local_1[gi] = ci
    end

    akk_entries = Dict{Tuple{Int,Int}, Nothing}()
    for (_, li, lj) in hess_per_scenario_diag[1]
        akk_entries[(li, lj)] = nothing
    end
    for i in 1:nv
        akk_entries[(i, i)] = nothing
    end
    for gi in eq_cons_1
        for (_, col) in get(jac_by_constraint, gi, Tuple{Int,Int}[])
            if col >= vr_start_1 && col < vr_start_1 + nv
                akk_entries[(nv + eq_local_1[gi], col - vr_start_1 + 1)] = nothing
            end
        end
    end
    for (ci, _) in enumerate(eq_cons_1)
        akk_entries[(nv + ci, nv + ci)] = nothing
    end
    for gi in ineq_cons_1
        local_vars = Int[]
        for (_, col) in get(jac_by_constraint, gi, Tuple{Int,Int}[])
            if col >= vr_start_1 && col < vr_start_1 + nv
                push!(local_vars, col - vr_start_1 + 1)
            end
        end
        for a in local_vars, b in local_vars
            a >= b && (akk_entries[(a, b)] = nothing)
        end
    end

    akk_nnz = length(akk_entries)
    akk_I = Vector{Int32}(undef, akk_nnz)
    akk_J = Vector{Int32}(undef, akk_nnz)
    akk_V = zeros(T, akk_nnz)
    for (idx, ((ri, rj), _)) in enumerate(akk_entries)
        akk_I[idx] = Int32(ri)
        akk_J[idx] = Int32(rj)
    end

    akk_coo_cpu = MadNLP.SparseMatrixCOO(blk_size, blk_size, akk_I, akk_J, akk_V)
    akk_csc_cpu, _ = MadNLP.coo_to_csc(akk_coo_cpu)

    akk_lookup = Dict{Tuple{Int,Int}, Int}()
    for col in 1:blk_size
        for p in akk_csc_cpu.colptr[col]:(akk_csc_cpu.colptr[col+1]-1)
            row = akk_csc_cpu.rowval[p]
            akk_lookup[(Int(row), Int(col))] = Int(p)
        end
    end

    nnz_per_scenario = length(akk_csc_cpu.nzval)

    # --- Build FLATTENED per-scenario index maps (all ns scenarios) ---
    # Each map type: concatenate scenario 1..ns entries
    all_hess_Akk_coo = Int[]
    all_hess_Akk_nzpos = Int[]
    all_hess_Cdk_coo = Int[]
    all_hess_Cdk_row = Int[]
    all_hess_Cdk_col = Int[]
    all_pr_global = Int[]
    all_pr_nzpos = Int[]
    all_du_global = Int[]
    all_du_nzpos = Int[]
    all_jeq_Akk_coo = Int[]
    all_jeq_Akk_nzpos = Int[]
    all_jeq_Cdk_coo = Int[]
    all_jeq_Cdk_row = Int[]
    all_jeq_Cdk_col = Int[]
    all_ineq_Akk_nzpos = Int[]
    all_ineq_Akk_jcoo1 = Int[]
    all_ineq_Akk_jcoo2 = Int[]
    all_ineq_Akk_bufidx = Int[]
    all_ineq_Cdk_row = Int[]
    all_ineq_Cdk_col = Int[]
    all_ineq_Cdk_jcoo_d = Int[]
    all_ineq_Cdk_jcoo_v = Int[]
    all_ineq_Cdk_bufidx = Int[]
    all_ineq_S_row = Int[]
    all_ineq_S_col = Int[]
    all_ineq_S_jcoo1 = Int[]
    all_ineq_S_jcoo2 = Int[]
    all_ineq_S_bufidx = Int[]

    n_per_s_hess_Akk = 0
    n_per_s_hess_Cdk = 0
    n_per_s_pr_diag = 0
    n_per_s_du_diag = 0
    n_per_s_jeq_Akk = 0
    n_per_s_jeq_Cdk = 0
    n_per_s_ineq_Akk = 0
    n_per_s_ineq_Cdk = 0
    n_per_s_ineq_S = 0

    eq_global_flat = Int[]

    for k in 1:ns
        vr_start = (k-1)*nv + 1
        eq_cons = eq_per_scenario[k]
        ineq_cons = ineq_per_scenario[k]

        eq_local = Dict{Int,Int}()
        for (ci, gi) in enumerate(eq_cons)
            eq_local[gi] = ci
        end

        # Hessian diagonal → A_kk
        for (ci, li, lj) in hess_per_scenario_diag[k]
            push!(all_hess_Akk_coo, ci)
            push!(all_hess_Akk_nzpos, akk_lookup[(li, lj)])
        end
        if k == 1
            n_per_s_hess_Akk = length(hess_per_scenario_diag[k])
        end

        # Hessian coupling → C_dk
        for (ci, di, vj) in hess_per_scenario_coupling[k]
            push!(all_hess_Cdk_coo, ci)
            push!(all_hess_Cdk_row, di)
            push!(all_hess_Cdk_col, vj)
        end
        if k == 1
            n_per_s_hess_Cdk = length(hess_per_scenario_coupling[k])
        end

        # pr_diag → A_kk
        for i in 1:nv
            push!(all_pr_global, vr_start + i - 1)
            push!(all_pr_nzpos, akk_lookup[(i, i)])
        end
        if k == 1
            n_per_s_pr_diag = nv
        end

        # du_diag → A_kk
        for (ci, gi) in enumerate(eq_cons)
            push!(all_du_global, gi)
            push!(all_du_nzpos, akk_lookup[(nv + ci, nv + ci)])
        end
        if k == 1
            n_per_s_du_diag = length(eq_cons)
        end

        # Equality Jacobian → A_kk and C_dk
        jeq_Akk_count = 0
        jeq_Cdk_count = 0
        for gi in eq_cons
            leq = eq_local[gi]
            for (coo_idx, col) in get(jac_by_constraint, gi, Tuple{Int,Int}[])
                if col >= vr_start && col < vr_start + nv
                    push!(all_jeq_Akk_coo, coo_idx)
                    push!(all_jeq_Akk_nzpos, akk_lookup[(nv + leq, col - vr_start + 1)])
                    jeq_Akk_count += 1
                elseif col >= d_start && col <= d_end
                    push!(all_jeq_Cdk_coo, coo_idx)
                    push!(all_jeq_Cdk_row, col - d_start + 1)
                    push!(all_jeq_Cdk_col, nv + leq)
                    jeq_Cdk_count += 1
                end
            end
        end
        if k == 1
            n_per_s_jeq_Akk = jeq_Akk_count
            n_per_s_jeq_Cdk = jeq_Cdk_count
        end

        # Inequality condensation
        ineq_Akk_count = 0
        ineq_Cdk_count = 0
        ineq_S_count = 0

        for gi in ineq_cons
            bidx = ineq_to_bufidx[gi]
            jac_entries = get(jac_by_constraint, gi, Tuple{Int,Int}[])

            v_entries = Tuple{Int,Int}[]
            d_entries = Tuple{Int,Int}[]
            for (coo_idx, col) in jac_entries
                if col >= vr_start && col < vr_start + nv
                    push!(v_entries, (coo_idx, col - vr_start + 1))
                elseif col >= d_start && col <= d_end
                    push!(d_entries, (coo_idx, col - d_start + 1))
                end
            end

            for (coo_a, la) in v_entries, (coo_b, lb) in v_entries
                if la >= lb
                    push!(all_ineq_Akk_nzpos, akk_lookup[(la, lb)])
                    push!(all_ineq_Akk_jcoo1, coo_a)
                    push!(all_ineq_Akk_jcoo2, coo_b)
                    push!(all_ineq_Akk_bufidx, bidx)
                    ineq_Akk_count += 1
                end
            end

            for (coo_d, di) in d_entries, (coo_v, lv) in v_entries
                push!(all_ineq_Cdk_row, di)
                push!(all_ineq_Cdk_col, lv)
                push!(all_ineq_Cdk_jcoo_d, coo_d)
                push!(all_ineq_Cdk_jcoo_v, coo_v)
                push!(all_ineq_Cdk_bufidx, bidx)
                ineq_Cdk_count += 1
            end

            for (coo_a, da) in d_entries, (coo_b, db) in d_entries
                push!(all_ineq_S_row, da)
                push!(all_ineq_S_col, db)
                push!(all_ineq_S_jcoo1, coo_a)
                push!(all_ineq_S_jcoo2, coo_b)
                push!(all_ineq_S_bufidx, bidx)
                ineq_S_count += 1
            end
        end
        if k == 1
            n_per_s_ineq_Akk = ineq_Akk_count
            n_per_s_ineq_Cdk = ineq_Cdk_count
            n_per_s_ineq_S = ineq_S_count
        end

        # Eq global indices
        append!(eq_global_flat, eq_cons)
    end

    # --- Create batched CSC on GPU ---
    batched_colPtr = CuVector{Cint}(Vector{Cint}(akk_csc_cpu.colptr))
    batched_rowVal = CuVector{Cint}(Vector{Cint}(akk_csc_cpu.rowval))
    batched_nzVal = CUDA.fill(zero(T), ns * nnz_per_scenario)

    A_kk_batched = CUSPARSE.CuSparseMatrixCSC{T, Cint}(
        batched_colPtr, batched_rowVal, batched_nzVal, (blk_size, blk_size),
    )

    # --- Dense arrays on GPU ---
    aug_com = CuMatrix{T}(undef, nd, nd)
    fill!(aug_com, zero(T))
    C_dk_batched = CuArray{T, 3}(undef, nd, blk_size, ns)
    fill!(C_dk_batched, zero(T))
    tmp_blk_nd_batched = CuArray{T, 3}(undef, blk_size, nd, ns)
    S_contrib = CuArray{T, 3}(undef, nd, nd, ns)

    # --- Diagonal vectors on GPU ---
    reg     = CuVector{T}(undef, n + ns_ineq)
    pr_diag = CuVector{T}(undef, n + ns_ineq)
    du_diag = CuVector{T}(undef, m)
    l_diag  = CUDA.fill(one(T), nlb)
    u_diag  = CUDA.fill(one(T), nub)
    l_lower = CUDA.fill(zero(T), nlb)
    u_lower = CUDA.fill(zero(T), nub)

    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    # --- Buffers ---
    diag_buffer = CuVector{T}(undef, max(ns_ineq, 1))
    buffer      = CuVector{T}(undef, m)
    rhs_d       = CuVector{T}(undef, nd)
    rhs_k_batched = CuMatrix{T}(undef, blk_size, ns)
    solve_buffer = CuVector{T}(undef, blk_size * ns)

    # --- Transfer flattened index maps to GPU ---
    gpu_hess_Akk_coo = CuVector{Int}(all_hess_Akk_coo)
    gpu_hess_Akk_nzpos = CuVector{Int}(all_hess_Akk_nzpos)
    gpu_hess_Cdk_coo = CuVector{Int}(all_hess_Cdk_coo)
    gpu_hess_Cdk_row = CuVector{Int}(all_hess_Cdk_row)
    gpu_hess_Cdk_col = CuVector{Int}(all_hess_Cdk_col)
    gpu_pr_diag_global = CuVector{Int}(all_pr_global)
    gpu_pr_diag_nzpos = CuVector{Int}(all_pr_nzpos)
    gpu_du_diag_global = CuVector{Int}(all_du_global)
    gpu_du_diag_nzpos = CuVector{Int}(all_du_nzpos)
    gpu_jeq_Akk_coo = CuVector{Int}(all_jeq_Akk_coo)
    gpu_jeq_Akk_nzpos = CuVector{Int}(all_jeq_Akk_nzpos)
    gpu_jeq_Cdk_coo = CuVector{Int}(all_jeq_Cdk_coo)
    gpu_jeq_Cdk_row = CuVector{Int}(all_jeq_Cdk_row)
    gpu_jeq_Cdk_col = CuVector{Int}(all_jeq_Cdk_col)
    gpu_ineq_Akk_nzpos = CuVector{Int}(all_ineq_Akk_nzpos)
    gpu_ineq_Akk_jcoo1 = CuVector{Int}(all_ineq_Akk_jcoo1)
    gpu_ineq_Akk_jcoo2 = CuVector{Int}(all_ineq_Akk_jcoo2)
    gpu_ineq_Akk_bufidx = CuVector{Int}(all_ineq_Akk_bufidx)
    gpu_ineq_Cdk_row = CuVector{Int}(all_ineq_Cdk_row)
    gpu_ineq_Cdk_col = CuVector{Int}(all_ineq_Cdk_col)
    gpu_ineq_Cdk_jcoo_d = CuVector{Int}(all_ineq_Cdk_jcoo_d)
    gpu_ineq_Cdk_jcoo_v = CuVector{Int}(all_ineq_Cdk_jcoo_v)
    gpu_ineq_Cdk_bufidx = CuVector{Int}(all_ineq_Cdk_bufidx)
    gpu_ineq_S_row = CuVector{Int}(all_ineq_S_row)
    gpu_ineq_S_col = CuVector{Int}(all_ineq_S_col)
    gpu_ineq_S_jcoo1 = CuVector{Int}(all_ineq_S_jcoo1)
    gpu_ineq_S_jcoo2 = CuVector{Int}(all_ineq_S_jcoo2)
    gpu_ineq_S_bufidx = CuVector{Int}(all_ineq_S_bufidx)
    gpu_hess_S_coo = CuVector{Int}(hess_S_coo_list)
    gpu_hess_S_row = CuVector{Int}(hess_S_row_list)
    gpu_hess_S_col = CuVector{Int}(hess_S_col_list)
    gpu_eq_global_indices = CuVector{Int}(eq_global_flat)

    # --- Create solvers ---
    quasi_newton = MadNLP.create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    scenario_solver = schur_scenario_linear_solver(A_kk_batched)
    _linear_solver = linear_solver(aug_com; opt=opt_linear_solver)

    return GPUSchurComplementKKTSystem(
        hess, jac,
        hess_raw, jt_coo,
        hess_csc, hess_csc_map, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ns, nv, nd, nc, nc_eq_per_s, nc_ineq_per_s, blk_size,
        A_kk_batched, nnz_per_scenario,
        C_dk_batched, aug_com,
        diag_buffer, buffer, rhs_d, rhs_k_batched, tmp_blk_nd_batched, S_contrib, solve_buffer,
        n_per_s_hess_Akk, gpu_hess_Akk_coo, gpu_hess_Akk_nzpos,
        n_per_s_hess_Cdk, gpu_hess_Cdk_coo, gpu_hess_Cdk_row, gpu_hess_Cdk_col,
        n_per_s_pr_diag, gpu_pr_diag_global, gpu_pr_diag_nzpos,
        n_per_s_du_diag, gpu_du_diag_global, gpu_du_diag_nzpos,
        n_per_s_jeq_Akk, gpu_jeq_Akk_coo, gpu_jeq_Akk_nzpos,
        n_per_s_jeq_Cdk, gpu_jeq_Cdk_coo, gpu_jeq_Cdk_row, gpu_jeq_Cdk_col,
        n_per_s_ineq_Akk, gpu_ineq_Akk_nzpos, gpu_ineq_Akk_jcoo1, gpu_ineq_Akk_jcoo2, gpu_ineq_Akk_bufidx,
        n_per_s_ineq_Cdk, gpu_ineq_Cdk_row, gpu_ineq_Cdk_col, gpu_ineq_Cdk_jcoo_d, gpu_ineq_Cdk_jcoo_v, gpu_ineq_Cdk_bufidx,
        n_per_s_ineq_S, gpu_ineq_S_row, gpu_ineq_S_col, gpu_ineq_S_jcoo1, gpu_ineq_S_jcoo2, gpu_ineq_S_bufidx,
        gpu_hess_S_coo, gpu_hess_S_row, gpu_hess_S_col,
        gpu_eq_global_indices,
        n_eq_total, CuVector{Int}(cpu_ind_eq),
        ns_ineq, CuVector{Int}(cpu_ind_ineq), CuVector{Int}(cpu_ind_lb), CuVector{Int}(cpu_ind_ub),
        scenario_solver, _linear_solver,
        Dict{Symbol, Any}(),
    )
end

# --- Trivial accessors ---
MadNLP.num_variables(kkt::GPUSchurComplementKKTSystem) = size(kkt.hess_csc, 1)

function MadNLP.get_slack_regularization(kkt::GPUSchurComplementKKTSystem)
    n = MadNLP.num_variables(kkt)
    return view(kkt.pr_diag, n+1:n+kkt.n_ineq)
end

function MadNLP.is_inertia_correct(kkt::GPUSchurComplementKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0 && num_neg == 0)
end

MadNLP.nnz_jacobian(kkt::GPUSchurComplementKKTSystem) = MadNLP.nnz(kkt.jt_coo)

function MadNLP.jtprod!(y::VT, kkt::GPUSchurComplementKKTSystem, x::VT) where {VT <: CuVector}
    nx = MadNLP.num_variables(kkt)
    ns_ineq = kkt.n_ineq
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns_ineq)
    mul!(yx, kkt.jt_csc, x)
    ys .= -@view(x[kkt.ind_ineq])
    return
end

function MadNLP.compress_jacobian!(kkt::GPUSchurComplementKKTSystem)
    MadNLP.transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function MadNLP.compress_hessian!(kkt::GPUSchurComplementKKTSystem)
    MadNLP.transfer!(kkt.hess_csc, kkt.hess_raw, kkt.hess_csc_map)
end

# --- build_kkt! ---
function MadNLP.build_kkt!(kkt::GPUSchurComplementKKTSystem{T}) where T
    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    n = MadNLP.num_variables(kkt)
    blk = kkt.blk_size
    backend = CUDABackend()
    nzval = kkt.A_kk_batched.nzVal
    nnz_s = kkt.nnz_per_scenario

    # Compute condensing diagonal for inequalities
    if kkt.n_ineq > 0
        Sigma_s = view(kkt.pr_diag, n+1:n+kkt.n_ineq)
        Sigma_d = @view(kkt.du_diag[kkt.ind_ineq])
        kkt.diag_buffer .= Sigma_s ./ (one(T) .- Sigma_d .* Sigma_s)
    end

    # Zero out assembly targets
    fill!(kkt.aug_com, zero(T))
    fill!(nzval, zero(T))
    fill!(kkt.C_dk_batched, zero(T))

    # Initialize S from design-design Hessian
    n_hess_S = length(kkt.hess_S_coo)
    if n_hess_S > 0
        _init_S_hess_kernel!(backend)(
            kkt.aug_com, kkt.hess, kkt.hess_S_coo, kkt.hess_S_row, kkt.hess_S_col;
            ndrange=n_hess_S,
        )
    end
    if nd > 0
        _init_S_diag_kernel!(backend)(kkt.aug_com, kkt.pr_diag, ns * nv, nd; ndrange=nd)
    end

    # Scatter Hessian diagonal → A_kk (flattened maps)
    n_hA = kkt.n_per_s_hess_Akk
    if n_hA > 0
        _scatter_hess_Akk_kernel!(backend)(
            nzval, kkt.hess, kkt.gpu_hess_Akk_coo, kkt.gpu_hess_Akk_nzpos,
            nnz_s, n_hA;
            ndrange=ns * n_hA,
        )
    end

    # Scatter Hessian coupling → C_dk
    n_hC = kkt.n_per_s_hess_Cdk
    if n_hC > 0
        _scatter_hess_Cdk_kernel!(backend)(
            kkt.C_dk_batched, kkt.hess, kkt.gpu_hess_Cdk_coo,
            kkt.gpu_hess_Cdk_row, kkt.gpu_hess_Cdk_col, n_hC;
            ndrange=ns * n_hC,
        )
    end

    # Scatter pr_diag → A_kk diagonal
    n_pr = kkt.n_per_s_pr_diag
    if n_pr > 0
        _scatter_diag_Akk_kernel!(backend)(
            nzval, kkt.pr_diag, kkt.gpu_pr_diag_global, kkt.gpu_pr_diag_nzpos,
            nnz_s, n_pr;
            ndrange=ns * n_pr,
        )
    end

    # Scatter du_diag → A_kk diagonal
    n_du = kkt.n_per_s_du_diag
    if n_du > 0
        _scatter_diag_Akk_kernel!(backend)(
            nzval, kkt.du_diag, kkt.gpu_du_diag_global, kkt.gpu_du_diag_nzpos,
            nnz_s, n_du;
            ndrange=ns * n_du,
        )
    end

    # Scatter equality Jacobian → A_kk
    n_jA = kkt.n_per_s_jeq_Akk
    if n_jA > 0
        _scatter_jeq_Akk_kernel!(backend)(
            nzval, kkt.jac, kkt.gpu_jeq_Akk_coo, kkt.gpu_jeq_Akk_nzpos,
            nnz_s, n_jA;
            ndrange=ns * n_jA,
        )
    end

    # Scatter equality Jacobian coupling → C_dk
    n_jC = kkt.n_per_s_jeq_Cdk
    if n_jC > 0
        _scatter_jeq_Cdk_kernel!(backend)(
            kkt.C_dk_batched, kkt.jac, kkt.gpu_jeq_Cdk_coo,
            kkt.gpu_jeq_Cdk_row, kkt.gpu_jeq_Cdk_col, n_jC;
            ndrange=ns * n_jC,
        )
    end

    # Inequality condensation → A_kk
    n_iA = kkt.n_per_s_ineq_Akk
    if n_iA > 0
        _ineq_condense_Akk_kernel!(backend)(
            nzval, kkt.jac, kkt.diag_buffer,
            kkt.gpu_ineq_Akk_nzpos, kkt.gpu_ineq_Akk_jcoo1,
            kkt.gpu_ineq_Akk_jcoo2, kkt.gpu_ineq_Akk_bufidx,
            nnz_s, n_iA;
            ndrange=ns * n_iA,
        )
    end

    # Inequality condensation → C_dk
    n_iC = kkt.n_per_s_ineq_Cdk
    if n_iC > 0
        _ineq_condense_Cdk_kernel!(backend)(
            kkt.C_dk_batched, kkt.jac, kkt.diag_buffer,
            kkt.gpu_ineq_Cdk_row, kkt.gpu_ineq_Cdk_col,
            kkt.gpu_ineq_Cdk_jcoo_d, kkt.gpu_ineq_Cdk_jcoo_v,
            kkt.gpu_ineq_Cdk_bufidx, n_iC;
            ndrange=ns * n_iC,
        )
    end

    # Inequality condensation → S (atomic adds)
    n_iS = kkt.n_per_s_ineq_S
    if n_iS > 0
        _ineq_condense_S_kernel!(backend)(
            kkt.aug_com, kkt.jac, kkt.diag_buffer,
            kkt.gpu_ineq_S_row, kkt.gpu_ineq_S_col,
            kkt.gpu_ineq_S_jcoo1, kkt.gpu_ineq_S_jcoo2,
            kkt.gpu_ineq_S_bufidx, n_iS, ns;
            ndrange=ns * n_iS,
        )
    end

    CUDA.synchronize()

    # Factorize all scenario blocks in one batched cuDSS call
    MadNLP.factorize!(kkt.scenario_solver)

    # Compute tmp = A_kk^{-1} * C_dk' column by column (nd batched solves)
    for j in 1:nd
        # C_dk'[:,j] for all k: extract C_dk[j, :, k] → solve_buffer columns
        for k_idx in 1:ns
            src = view(kkt.C_dk_batched, j, :, k_idx)
            dst = view(kkt.solve_buffer, (k_idx-1)*blk + 1 : k_idx*blk)
            copyto!(dst, src)
        end
        MadNLP.solve_linear_system!(kkt.scenario_solver, kkt.solve_buffer)
        for k_idx in 1:ns
            src = view(kkt.solve_buffer, (k_idx-1)*blk + 1 : k_idx*blk)
            dst = view(kkt.tmp_blk_nd_batched, :, j, k_idx)
            copyto!(dst, src)
        end
    end

    # S -= C_dk * tmp_blk_nd via strided batched GEMM
    CUBLAS.gemm_strided_batched!('N', 'N', -one(T),
        kkt.C_dk_batched, kkt.tmp_blk_nd_batched, zero(T), kkt.S_contrib)

    # Accumulate: S += sum(S_contrib, dims=3)
    S_sum = dropdims(sum(kkt.S_contrib; dims=3); dims=3)
    kkt.aug_com .+= S_sum

    return
end

# --- factorize_kkt! ---
function MadNLP.factorize_kkt!(kkt::GPUSchurComplementKKTSystem)
    return MadNLP.factorize!(kkt.linear_solver)
end

# --- solve_kkt! ---
function MadNLP.solve_kkt!(
    kkt::GPUSchurComplementKKTSystem{T},
    w::MadNLP.AbstractKKTVector{T},
) where T

    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    n = MadNLP.num_variables(kkt)
    blk = kkt.blk_size
    nc_eq = kkt.nc_eq_per_s
    backend = CUDABackend()

    wx = view(MadNLP.full(w), 1:n)
    ws = view(MadNLP.full(w), n+1:n+kkt.n_ineq)
    wy = MadNLP.dual(w)

    Sigma_s = MadNLP.get_slack_regularization(kkt)

    MadNLP.reduce_rhs!(kkt, w)

    # Step 1: condense inequality contributions
    fill!(kkt.buffer, zero(T))
    if kkt.n_ineq > 0
        kkt.buffer[kkt.ind_ineq] .= kkt.diag_buffer .* (wy[kkt.ind_ineq] .+ ws ./ Sigma_s)
        mul!(wx, kkt.jt_csc, kkt.buffer, one(T), one(T))
    end

    # Step 2: Extract per-scenario RHS blocks via kernel
    if blk * ns > 0
        _extract_rhs_kernel!(backend)(
            kkt.rhs_k_batched, wx, wy, kkt.eq_global_indices,
            nv, nc_eq, ns, blk;
            ndrange=blk * ns,
        )
        CUDA.synchronize()
    end
    copyto!(kkt.rhs_d, view(wx, ns*nv+1:ns*nv+nd))

    # Step 3: Forward elimination — batched solve
    copyto!(kkt.solve_buffer, vec(kkt.rhs_k_batched))
    MadNLP.solve_linear_system!(kkt.scenario_solver, kkt.solve_buffer)
    copyto!(vec(kkt.rhs_k_batched), kkt.solve_buffer)

    # rhs_d -= sum_k C_dk * rhs_k
    for k in 1:ns
        C_k = view(kkt.C_dk_batched, :, :, k)
        rhs_k = view(kkt.rhs_k_batched, :, k)
        mul!(kkt.rhs_d, C_k, rhs_k, -one(T), one(T))
    end

    # Step 4: Solve Schur complement
    MadNLP.solve_linear_system!(kkt.linear_solver, kkt.rhs_d)

    # Step 5: Back-substitution
    for k in 1:ns
        tmp_k = view(kkt.tmp_blk_nd_batched, :, :, k)
        rhs_k = view(kkt.rhs_k_batched, :, k)
        mul!(rhs_k, tmp_k, kkt.rhs_d, -one(T), one(T))
    end

    # Step 6: Write back to w via kernel
    if blk * ns > 0
        _writeback_rhs_kernel!(backend)(
            wx, wy, kkt.rhs_k_batched, kkt.eq_global_indices,
            nv, nc_eq, ns, blk;
            ndrange=blk * ns,
        )
        CUDA.synchronize()
    end
    copyto!(view(wx, ns*nv+1:ns*nv+nd), kkt.rhs_d)

    # Step 7: Recover inequality duals and slacks
    if kkt.n_ineq > 0
        # Save equality duals
        eq_backup = similar(wy, nc_eq * ns)
        flat_idx = 1
        for k in 1:ns
            for eq_i in 1:nc_eq
                gi_idx = (k-1)*nc_eq + eq_i
                CUDA.@allowscalar eq_backup[flat_idx] = wy[kkt.eq_global_indices[gi_idx]]
                flat_idx += 1
            end
        end

        # J * Δx
        mul!(wy, kkt.jt_csc', wx)

        # Restore equality duals
        flat_idx = 1
        for k in 1:ns
            for eq_i in 1:nc_eq
                gi_idx = (k-1)*nc_eq + eq_i
                CUDA.@allowscalar wy[kkt.eq_global_indices[gi_idx]] = eq_backup[flat_idx]
                flat_idx += 1
            end
        end

        # Inequality dual recovery
        wy[kkt.ind_ineq] .= kkt.diag_buffer .* wy[kkt.ind_ineq] .- kkt.buffer[kkt.ind_ineq]
        ws .= (ws .+ view(wy, kkt.ind_ineq)) ./ Sigma_s
    end

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

# --- mul! for iterative refinement ---
function MadNLP.mul!(
    w::MadNLP.AbstractKKTVector{T, VT},
    kkt::GPUSchurComplementKKTSystem{T},
    x::MadNLP.AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT <: CuVector{T}}
    n = MadNLP.num_variables(kkt)

    wx = @view(MadNLP.primal(w)[1:n])
    ws = @view(MadNLP.primal(w)[n+1:end])
    wy = MadNLP.dual(w)

    xx = @view(MadNLP.primal(x)[1:n])
    xs = @view(MadNLP.primal(x)[n+1:end])
    xz = @view(MadNLP.dual(x)[kkt.ind_ineq])

    wx .= beta .* wx
    mul!(wx, Symmetric(kkt.hess_csc, :L), xx, alpha, one(T))

    m = size(kkt.jt_csc, 2)
    if m > 0
        mul!(wx, kkt.jt_csc, MadNLP.dual(x), alpha, one(T))
        mul!(wy, kkt.jt_csc', xx, alpha, beta)
    else
        wy .= beta .* wy
    end
    ws .= beta .* ws .- alpha .* xz
    @view(MadNLP.dual(w)[kkt.ind_ineq]) .-= alpha .* xs
    MadNLP._kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function MadNLP.mul_hess_blk!(wx::VT, kkt::GPUSchurComplementKKTSystem{T}, t) where {T, VT <: CuVector{T}}
    n = MadNLP.num_variables(kkt)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_csc, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
