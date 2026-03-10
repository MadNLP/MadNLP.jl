
"""
    ScenarioBlockMap

Precomputed index mappings for one scenario — maps from global COO indices
to per-block positions in A_kk (sparse), C_dk (dense), and S (dense).
"""
struct ScenarioBlockMap
    # Hessian diagonal block: global hess COO index → A_kk nzval position
    hess_Akk_coo::Vector{Int}
    hess_Akk_nzpos::Vector{Int}

    # Hessian coupling block: global hess COO index → (C_dk row, C_dk col)
    hess_Cdk_coo::Vector{Int}
    hess_Cdk_row::Vector{Int}
    hess_Cdk_col::Vector{Int}

    # Equality Jacobian, scenario vars → A_kk lower triangle only
    jeq_Akk_coo::Vector{Int}
    jeq_Akk_nzpos::Vector{Int}

    # Equality Jacobian, design vars → C_dk
    jeq_Cdk_coo::Vector{Int}
    jeq_Cdk_row::Vector{Int}   # design var index (1:nd)
    jeq_Cdk_col::Vector{Int}   # nv + eq_local_idx

    # Inequality condensation → A_kk (lower triangle)
    ineq_Akk_nzpos::Vector{Int}
    ineq_Akk_jcoo1::Vector{Int}
    ineq_Akk_jcoo2::Vector{Int}
    ineq_Akk_bufidx::Vector{Int}

    # Inequality condensation → C_dk
    ineq_Cdk_row::Vector{Int}     # design var index (1:nd)
    ineq_Cdk_col::Vector{Int}     # scenario local var (1:nv)
    ineq_Cdk_jcoo_d::Vector{Int}  # jac COO for design var
    ineq_Cdk_jcoo_v::Vector{Int}  # jac COO for scenario var
    ineq_Cdk_bufidx::Vector{Int}

    # Inequality condensation → S (full, both triangles)
    ineq_S_row::Vector{Int}
    ineq_S_col::Vector{Int}
    ineq_S_jcoo1::Vector{Int}
    ineq_S_jcoo2::Vector{Int}
    ineq_S_bufidx::Vector{Int}

    # Diagonal positions in A_kk nzval for pr_diag (nv entries)
    pr_diag_global::Vector{Int}
    pr_diag_nzpos::Vector{Int}

    # Diagonal positions in A_kk nzval for du_diag (nc_eq entries)
    du_diag_global::Vector{Int}
    du_diag_nzpos::Vector{Int}
end

"""
    SchurComplementKKTSystem{T, VT, MT, QN, LS, LS2, VI} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

KKT system exploiting block-arrowhead structure from two-stage stochastic programs
via Schur complement decomposition, using sparse COO/CSC storage for the global
Hessian and Jacobian and sparse per-scenario block solvers.

Variable layout: `[v_1, ..., v_ns, d]` where `v_k ∈ R^nv`, `d ∈ R^nd`.
Constraint layout: `[c_1, ..., c_ns]` where `c_k ∈ R^nc`.

The augmented per-scenario block `A_k` (size `blk_size × blk_size`) is stored
as a sparse lower-triangular `SparseMatrixCSC` and factored by a configurable
sparse solver (default `LDLSolver`). The coupling blocks `C_dk` remain dense.
The Schur complement `S = aug_com` (size `nd × nd`) is dense.
"""
struct SchurComplementKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,
    LS2,
    VI <: AbstractVector{Int}
    } <: AbstractCondensedKKTSystem{T, VT, MT, QN}

    # COO value buffers (filled by NLP callback via jac_coord!/hess_coord!)
    hess::VT                        # length cb.nnzh
    jac::VT                         # length cb.nnzj

    # COO structures (share value buffers above)
    hess_raw::SparseMatrixCOO{T, Int32, VT, Vector{Int32}}
    jt_coo::SparseMatrixCOO{T, Int32, VT, Vector{Int32}}

    # CSC representations (for jtprod, mul, dual recovery)
    hess_csc::SparseMatrixCSC{T, Int32}
    hess_csc_map::Vector{Int}
    jt_csc::SparseMatrixCSC{T, Int32}
    jt_csc_map::Vector{Int}

    quasi_newton::QN

    # Standard MadNLP diagonal vectors
    reg::VT                         # n_total + n_ineq
    pr_diag::VT                     # n_total + n_ineq
    du_diag::VT                     # m_total
    l_diag::VT                      # nlb
    u_diag::VT                      # nub
    l_lower::VT                     # nlb
    u_lower::VT                     # nub

    # Two-stage dimensions
    ns::Int
    nv::Int
    nd::Int
    nc::Int
    nc_eq_per_s::Int
    nc_ineq_per_s::Int
    blk_size::Int                   # = nv + nc_eq_per_s

    # Per-scenario sparse augmented blocks (lower triangle only)
    A_kk::Vector{SparseMatrixCSC{T, Int32}}
    C_dk::Vector{MT}                # ns × (nd × blk_size) — dense

    # Schur complement (what the dense linear solver sees)
    aug_com::MT                     # nd × nd

    # Buffers
    diag_buffer::VT                 # n_ineq — condensing diagonal
    buffer::VT                      # m_total — general
    rhs_d::VT                       # nd — design RHS
    rhs_k::Vector{VT}              # ns × blk_size — scenario RHS buffers
    tmp_blk_nd::Vector{MT}         # ns × (blk_size × nd)
    solve_buffers::Vector{VT}      # ns × blk_size — per-scenario column-by-column solve buffers

    # Precomputed index maps
    block_maps::Vector{ScenarioBlockMap}
    hess_S_coo::Vector{Int}         # COO indices for design-design Hessian
    hess_S_row::Vector{Int}         # S row (1:nd)
    hess_S_col::Vector{Int}         # S col (1:nd)

    # Scenario classification
    eq_per_scenario::Vector{Vector{Int}}
    ineq_per_scenario::Vector{Vector{Int}}

    # Inequality/equality/bound index info
    n_eq::Int
    ind_eq::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # Solvers
    scenario_solvers::Vector{LS2}
    linear_solver::LS               # for Schur complement S (dense)
    etc::Dict{Symbol, Any}
end

# --- Helper: find index of gi in ind_ineq → diag_buffer index ---
function _ineq_buf_idx(ind_ineq, gi::Int)
    @inbounds for idx in 1:length(ind_ineq)
        if ind_ineq[idx] == gi
            return idx
        end
    end
    return 0
end

function create_kkt_system(
    ::Type{SchurComplementKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
    schur_ns::Int=0,
    schur_nv::Int=0,
    schur_nd::Int=0,
    schur_nc::Int=0,
    schur_scenario_linear_solver::Type=LDLSolver,
) where {T, VT}

    n = cb.nvar
    m = cb.ncon
    ns_ineq = length(cb.ind_ineq)
    n_eq = m - ns_ineq
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    # Auto-detect dimensions from TwoStageTags if not provided
    if schur_ns == 0 && hasproperty(cb.nlp, :tags)
        tags = cb.nlp.tags
        if hasproperty(tags, :ns) && hasproperty(tags, :var_scenario) && hasproperty(tags, :con_scenario)
            schur_ns = tags.ns
            var_scen = Array(tags.var_scenario)
            con_scen = Array(tags.con_scenario)
            schur_nd = count(==(0), var_scen)
            schur_nv = count(==(1), var_scen)
            schur_nc = count(==(1), con_scen)
        end
    end

    @assert schur_ns > 0 "schur_ns must be specified and positive (or use TwoStageTags for auto-detection)"
    @assert schur_nv > 0 "schur_nv must be specified and positive"
    @assert schur_nd > 0 "schur_nd must be specified and positive"
    @assert n == schur_ns * schur_nv + schur_nd "Variable count mismatch: n=$n != ns*nv+nd=$(schur_ns*schur_nv+schur_nd)"
    @assert m == schur_ns * schur_nc "Constraint count mismatch: m=$m != ns*nc=$(schur_ns*schur_nc)"

    ns = schur_ns
    nv = schur_nv
    nd = schur_nd
    nc = schur_nc

    # --- Get sparsity patterns ---
    jac_sparsity_I = Vector{Int32}(undef, cb.nnzj)
    jac_sparsity_J = Vector{Int32}(undef, cb.nnzj)
    _jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, hessian_approximation)
    force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    n_hess = length(hess_sparsity_I)
    n_jac = length(jac_sparsity_I)

    # --- COO value buffers ---
    hess = VT(undef, n_hess)
    jac = VT(undef, n_jac)
    fill!(hess, zero(T))
    fill!(jac, zero(T))

    # --- Build global COO + CSC ---
    hess_raw = SparseMatrixCOO(n, n, hess_sparsity_I, hess_sparsity_J, hess)
    jt_coo = SparseMatrixCOO(n, m, jac_sparsity_J, jac_sparsity_I, jac)  # transposed

    hess_csc, hess_csc_map = coo_to_csc(hess_raw)
    jt_csc, jt_csc_map = coo_to_csc(jt_coo)

    # --- Classify constraints per scenario ---
    ind_eq_set = Set(cb.ind_eq)
    ind_ineq_set = Set(cb.ind_ineq)

    eq_per_scenario = Vector{Vector{Int}}(undef, ns)
    ineq_per_scenario = Vector{Vector{Int}}(undef, ns)

    nc_eq_per_s = 0
    nc_ineq_per_s = 0
    for k in 1:ns
        cr = (k-1)*nc+1 : k*nc
        eq_per_scenario[k] = Int[]
        ineq_per_scenario[k] = Int[]
        for gi in cr
            if gi in ind_eq_set
                push!(eq_per_scenario[k], gi)
            end
            if gi in ind_ineq_set
                push!(ineq_per_scenario[k], gi)
            end
        end
        if k == 1
            nc_eq_per_s = length(eq_per_scenario[k])
            nc_ineq_per_s = length(ineq_per_scenario[k])
        end
    end

    blk_size = nv + nc_eq_per_s

    # --- Build index for quick ineq lookup ---
    ineq_to_bufidx = Dict{Int,Int}()
    for idx in 1:length(cb.ind_ineq)
        ineq_to_bufidx[cb.ind_ineq[idx]] = idx
    end

    # --- Build Jacobian COO lookup: for each constraint, which COO entries ---
    # jac_by_constraint[gi] = [(coo_idx, col), ...]
    jac_by_constraint = Dict{Int, Vector{Tuple{Int,Int}}}()
    for ci in 1:n_jac
        row = Int(jac_sparsity_I[ci])
        col = Int(jac_sparsity_J[ci])
        entries = get!(Vector{Tuple{Int,Int}}, jac_by_constraint, row)
        push!(entries, (ci, col))
    end

    # --- Precompute design variable range ---
    d_start = ns * nv + 1
    d_end = ns * nv + nd

    # --- Classify Hessian COO entries ---
    # hess_S_entries: design-design entries for Schur complement initialization
    hess_S_coo_list = Int[]
    hess_S_row_list = Int[]
    hess_S_col_list = Int[]

    # Per-scenario Hessian classification
    hess_per_scenario_diag = [Tuple{Int,Int,Int}[] for _ in 1:ns]   # (coo, local_i, local_j)
    hess_per_scenario_coupling = [Tuple{Int,Int,Int}[] for _ in 1:ns]  # (coo, design_local, var_local)

    for ci in 1:n_hess
        ri = Int(hess_sparsity_I[ci])
        rj = Int(hess_sparsity_J[ci])  # lower triangle: ri >= rj

        # Check if both are design vars
        if ri >= d_start && ri <= d_end && rj >= d_start && rj <= d_end
            di = ri - d_start + 1
            dj = rj - d_start + 1
            # Store both triangles for dense S
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

        # Check if one is design and one is scenario var
        # Lower triangle: ri >= rj. Design vars have larger indices.
        if ri >= d_start && ri <= d_end && rj < d_start
            di = ri - d_start + 1
            # Find which scenario rj belongs to
            k = div(rj - 1, nv) + 1
            if k >= 1 && k <= ns
                vj = rj - (k-1)*nv  # local var index
                push!(hess_per_scenario_coupling[k], (ci, di, vj))
            end
            continue
        end

        # Check if both are in the same scenario
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

    # --- Build per-scenario A_kk sparsity patterns and block maps ---
    A_kk_vec = Vector{SparseMatrixCSC{T, Int32}}(undef, ns)
    block_maps = Vector{ScenarioBlockMap}(undef, ns)

    for k in 1:ns
        vr_start = (k-1)*nv + 1
        eq_cons = eq_per_scenario[k]
        ineq_cons = ineq_per_scenario[k]

        # Local equation constraint index mapping: global → local eq index
        eq_local = Dict{Int,Int}()
        for (ci, gi) in enumerate(eq_cons)
            eq_local[gi] = ci
        end

        # Collect all lower-triangle (row, col) entries for A_kk
        # Using a dict to map (row,col) → list of sources
        akk_entries = Dict{Tuple{Int,Int}, Nothing}()

        # 1. Hessian diagonal entries (already lower triangle)
        for (_, li, lj) in hess_per_scenario_diag[k]
            akk_entries[(li, lj)] = nothing
        end

        # 2. Diagonal pr_diag entries
        for i in 1:nv
            akk_entries[(i, i)] = nothing
        end

        # 3. Equality Jacobian: row=nv+eq_local, col=var_local (lower triangle since nv+ci > j)
        for gi in eq_cons
            jac_entries = get(jac_by_constraint, gi, Tuple{Int,Int}[])
            for (_, col) in jac_entries
                if col >= vr_start && col < vr_start + nv
                    local_var = col - vr_start + 1
                    local_eq = eq_local[gi]
                    # A_kk row = nv + local_eq, col = local_var (always lower triangle)
                    akk_entries[(nv + local_eq, local_var)] = nothing
                end
            end
        end

        # 4. Diagonal du_diag entries
        for (ci, _) in enumerate(eq_cons)
            akk_entries[(nv + ci, nv + ci)] = nothing
        end

        # 5. Inequality condensation fill-in
        for gi in ineq_cons
            jac_entries = get(jac_by_constraint, gi, Tuple{Int,Int}[])
            # Collect scenario var entries for this constraint
            local_vars = Int[]
            for (_, col) in jac_entries
                if col >= vr_start && col < vr_start + nv
                    push!(local_vars, col - vr_start + 1)
                end
            end
            # All lower-triangle pairs
            for a in local_vars
                for b in local_vars
                    if a >= b
                        akk_entries[(a, b)] = nothing
                    end
                end
            end
        end

        # Build A_kk COO
        akk_nnz = length(akk_entries)
        akk_I = Vector{Int32}(undef, akk_nnz)
        akk_J = Vector{Int32}(undef, akk_nnz)
        akk_V = Vector{T}(undef, akk_nnz)
        fill!(akk_V, zero(T))

        for (idx, ((ri, rj), _)) in enumerate(akk_entries)
            akk_I[idx] = Int32(ri)
            akk_J[idx] = Int32(rj)
        end

        akk_coo = SparseMatrixCOO(blk_size, blk_size, akk_I, akk_J, akk_V)
        akk_csc, akk_csc_map = coo_to_csc(akk_coo)

        # Now we need a way to find nzval position for a given (row, col) in A_kk (lower triangle)
        # Build a lookup from (row, col) → nzval position in akk_csc
        akk_lookup = Dict{Tuple{Int,Int}, Int}()
        for col in 1:blk_size
            for p in akk_csc.colptr[col]:(akk_csc.colptr[col+1]-1)
                row = akk_csc.rowval[p]
                akk_lookup[(Int(row), Int(col))] = Int(p)
            end
        end

        # --- Build ScenarioBlockMap ---

        # Hessian diagonal → A_kk
        hess_Akk_coo_vec = Int[]
        hess_Akk_nzpos_vec = Int[]
        for (ci, li, lj) in hess_per_scenario_diag[k]
            nzpos = akk_lookup[(li, lj)]
            push!(hess_Akk_coo_vec, ci)
            push!(hess_Akk_nzpos_vec, nzpos)
        end

        # Hessian coupling → C_dk
        hess_Cdk_coo_vec = Int[]
        hess_Cdk_row_vec = Int[]
        hess_Cdk_col_vec = Int[]
        for (ci, di, vj) in hess_per_scenario_coupling[k]
            push!(hess_Cdk_coo_vec, ci)
            push!(hess_Cdk_row_vec, di)
            push!(hess_Cdk_col_vec, vj)
        end

        # Equality Jacobian → A_kk (lower triangle) and → C_dk
        jeq_Akk_coo_vec = Int[]
        jeq_Akk_nzpos_vec = Int[]
        jeq_Cdk_coo_vec = Int[]
        jeq_Cdk_row_vec = Int[]
        jeq_Cdk_col_vec = Int[]

        for gi in eq_cons
            local_eq = eq_local[gi]
            jac_entries = get(jac_by_constraint, gi, Tuple{Int,Int}[])
            for (coo_idx, col) in jac_entries
                if col >= vr_start && col < vr_start + nv
                    # Scenario var → A_kk lower triangle
                    local_var = col - vr_start + 1
                    nzpos = akk_lookup[(nv + local_eq, local_var)]
                    push!(jeq_Akk_coo_vec, coo_idx)
                    push!(jeq_Akk_nzpos_vec, nzpos)
                elseif col >= d_start && col <= d_end
                    # Design var → C_dk
                    di = col - d_start + 1
                    push!(jeq_Cdk_coo_vec, coo_idx)
                    push!(jeq_Cdk_row_vec, di)
                    push!(jeq_Cdk_col_vec, nv + local_eq)
                end
            end
        end

        # pr_diag → A_kk diagonal
        pr_diag_global_vec = Int[]
        pr_diag_nzpos_vec = Int[]
        for i in 1:nv
            gi = vr_start + i - 1
            nzpos = akk_lookup[(i, i)]
            push!(pr_diag_global_vec, gi)
            push!(pr_diag_nzpos_vec, nzpos)
        end

        # du_diag → A_kk diagonal
        du_diag_global_vec = Int[]
        du_diag_nzpos_vec = Int[]
        for (ci, gi) in enumerate(eq_cons)
            nzpos = akk_lookup[(nv + ci, nv + ci)]
            push!(du_diag_global_vec, gi)
            push!(du_diag_nzpos_vec, nzpos)
        end

        # Inequality condensation
        ineq_Akk_nzpos_vec = Int[]
        ineq_Akk_jcoo1_vec = Int[]
        ineq_Akk_jcoo2_vec = Int[]
        ineq_Akk_bufidx_vec = Int[]

        ineq_Cdk_row_vec = Int[]
        ineq_Cdk_col_vec = Int[]
        ineq_Cdk_jcoo_d_vec = Int[]
        ineq_Cdk_jcoo_v_vec = Int[]
        ineq_Cdk_bufidx_vec = Int[]

        ineq_S_row_vec = Int[]
        ineq_S_col_vec = Int[]
        ineq_S_jcoo1_vec = Int[]
        ineq_S_jcoo2_vec = Int[]
        ineq_S_bufidx_vec = Int[]

        for gi in ineq_cons
            bidx = ineq_to_bufidx[gi]
            jac_entries = get(jac_by_constraint, gi, Tuple{Int,Int}[])

            # Separate scenario var entries and design var entries
            v_entries = Tuple{Int,Int}[]  # (jac_coo_idx, local_var)
            d_entries = Tuple{Int,Int}[]  # (jac_coo_idx, design_local)

            for (coo_idx, col) in jac_entries
                if col >= vr_start && col < vr_start + nv
                    push!(v_entries, (coo_idx, col - vr_start + 1))
                elseif col >= d_start && col <= d_end
                    push!(d_entries, (coo_idx, col - d_start + 1))
                end
            end

            # A_kk condensation: lower-triangle pairs of v_entries
            for (coo_a, la) in v_entries
                for (coo_b, lb) in v_entries
                    if la >= lb
                        nzpos = akk_lookup[(la, lb)]
                        push!(ineq_Akk_nzpos_vec, nzpos)
                        push!(ineq_Akk_jcoo1_vec, coo_a)
                        push!(ineq_Akk_jcoo2_vec, coo_b)
                        push!(ineq_Akk_bufidx_vec, bidx)
                    end
                end
            end

            # C_dk condensation: design × scenario pairs
            for (coo_d, di) in d_entries
                for (coo_v, lv) in v_entries
                    push!(ineq_Cdk_row_vec, di)
                    push!(ineq_Cdk_col_vec, lv)
                    push!(ineq_Cdk_jcoo_d_vec, coo_d)
                    push!(ineq_Cdk_jcoo_v_vec, coo_v)
                    push!(ineq_Cdk_bufidx_vec, bidx)
                end
            end

            # S condensation: design × design pairs (full matrix)
            for (coo_a, da) in d_entries
                for (coo_b, db) in d_entries
                    push!(ineq_S_row_vec, da)
                    push!(ineq_S_col_vec, db)
                    push!(ineq_S_jcoo1_vec, coo_a)
                    push!(ineq_S_jcoo2_vec, coo_b)
                    push!(ineq_S_bufidx_vec, bidx)
                end
            end
        end

        block_maps[k] = ScenarioBlockMap(
            hess_Akk_coo_vec, hess_Akk_nzpos_vec,
            hess_Cdk_coo_vec, hess_Cdk_row_vec, hess_Cdk_col_vec,
            jeq_Akk_coo_vec, jeq_Akk_nzpos_vec,
            jeq_Cdk_coo_vec, jeq_Cdk_row_vec, jeq_Cdk_col_vec,
            ineq_Akk_nzpos_vec, ineq_Akk_jcoo1_vec, ineq_Akk_jcoo2_vec, ineq_Akk_bufidx_vec,
            ineq_Cdk_row_vec, ineq_Cdk_col_vec, ineq_Cdk_jcoo_d_vec, ineq_Cdk_jcoo_v_vec, ineq_Cdk_bufidx_vec,
            ineq_S_row_vec, ineq_S_col_vec, ineq_S_jcoo1_vec, ineq_S_jcoo2_vec, ineq_S_bufidx_vec,
            pr_diag_global_vec, pr_diag_nzpos_vec,
            du_diag_global_vec, du_diag_nzpos_vec,
        )

        A_kk_vec[k] = akk_csc
    end

    # --- Dense matrices ---
    aug_com = Matrix{T}(undef, nd, nd)
    C_dk = [Matrix{T}(undef, nd, blk_size) for _ in 1:ns]
    tmp_blk_nd = [Matrix{T}(undef, blk_size, nd) for _ in 1:ns]

    # --- Diagonal vectors ---
    reg     = VT(undef, n + ns_ineq)
    pr_diag = VT(undef, n + ns_ineq)
    du_diag = VT(undef, m)
    l_diag  = fill!(VT(undef, nlb), one(T))
    u_diag  = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))

    # --- Buffers ---
    diag_buffer = VT(undef, ns_ineq)
    buffer      = VT(undef, m)
    rhs_d       = VT(undef, nd)
    rhs_k       = [VT(undef, blk_size) for _ in 1:ns]
    solve_buffers = [VT(undef, blk_size) for _ in 1:ns]

    # --- Init ---
    fill!(aug_com, zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    # --- Create solvers ---
    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    scenario_solvers = [schur_scenario_linear_solver(A_kk_vec[k]) for k in 1:ns]
    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)

    return SchurComplementKKTSystem(
        hess, jac,
        hess_raw, jt_coo,
        hess_csc, hess_csc_map, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ns, nv, nd, nc,
        nc_eq_per_s, nc_ineq_per_s, blk_size,
        A_kk_vec, C_dk,
        aug_com,
        diag_buffer, buffer, rhs_d, rhs_k, tmp_blk_nd, solve_buffers,
        block_maps,
        hess_S_coo_list, hess_S_row_list, hess_S_col_list,
        eq_per_scenario, ineq_per_scenario,
        n_eq, cb.ind_eq,
        ns_ineq, cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        scenario_solvers,
        _linear_solver,
        Dict{Symbol, Any}(),
    )
end

num_variables(kkt::SchurComplementKKTSystem) = size(kkt.hess_csc, 1)

function get_slack_regularization(kkt::SchurComplementKKTSystem)
    n = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    return view(kkt.pr_diag, n+1:n+ns_ineq)
end

function is_inertia_correct(kkt::SchurComplementKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0 && num_neg == 0)
end

function jtprod!(y::AbstractVector, kkt::SchurComplementKKTSystem, x::AbstractVector)
    nx = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns_ineq)
    mul!(yx, kkt.jt_csc, x)
    ys .= -@view(x[kkt.ind_ineq])
    return
end

function compress_jacobian!(kkt::SchurComplementKKTSystem)
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function compress_hessian!(kkt::SchurComplementKKTSystem)
    transfer!(kkt.hess_csc, kkt.hess_raw, kkt.hess_csc_map)
end

nnz_jacobian(kkt::SchurComplementKKTSystem) = nnz(kkt.jt_coo)

function build_kkt!(kkt::SchurComplementKKTSystem{T, VT, MT}) where {T, VT, MT}
    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    n = num_variables(kkt)
    blk = kkt.blk_size

    # Compute condensing diagonal for inequalities
    if kkt.n_ineq > 0
        Sigma_s = view(kkt.pr_diag, n+1:n+kkt.n_ineq)
        Sigma_d = @view(kkt.du_diag[kkt.ind_ineq])
        kkt.diag_buffer .= Sigma_s ./ (one(T) .- Sigma_d .* Sigma_s)
    end

    # Initialize Schur complement S
    S = kkt.aug_com
    fill!(S, zero(T))

    # S += H_dd (from precomputed Hessian design-design entries)
    @inbounds for idx in 1:length(kkt.hess_S_coo)
        S[kkt.hess_S_row[idx], kkt.hess_S_col[idx]] += kkt.hess[kkt.hess_S_coo[idx]]
    end
    # S += pr_diag_dd
    @inbounds for i in 1:nd
        S[i, i] += kkt.pr_diag[ns*nv+i]
    end

    # Phase 1 (parallel): assemble per-scenario blocks, factorize, compute A_kk^{-1} * C_dk'
    @blas_safe_threads for k in 1:ns
        bm = kkt.block_maps[k]
        A_kk = kkt.A_kk[k]
        C_dk = kkt.C_dk[k]

        fill!(A_kk.nzval, zero(T))
        fill!(C_dk, zero(T))

        # Scatter Hessian diagonal entries → A_kk
        @inbounds for idx in 1:length(bm.hess_Akk_coo)
            A_kk.nzval[bm.hess_Akk_nzpos[idx]] += kkt.hess[bm.hess_Akk_coo[idx]]
        end

        # Scatter Hessian coupling entries → C_dk
        @inbounds for idx in 1:length(bm.hess_Cdk_coo)
            C_dk[bm.hess_Cdk_row[idx], bm.hess_Cdk_col[idx]] += kkt.hess[bm.hess_Cdk_coo[idx]]
        end

        # Add pr_diag to A_kk diagonal
        @inbounds for idx in 1:length(bm.pr_diag_global)
            A_kk.nzval[bm.pr_diag_nzpos[idx]] += kkt.pr_diag[bm.pr_diag_global[idx]]
        end

        # Add du_diag to A_kk diagonal
        @inbounds for idx in 1:length(bm.du_diag_global)
            A_kk.nzval[bm.du_diag_nzpos[idx]] += kkt.du_diag[bm.du_diag_global[idx]]
        end

        # Scatter equality Jacobian → A_kk (lower triangle only)
        @inbounds for idx in 1:length(bm.jeq_Akk_coo)
            A_kk.nzval[bm.jeq_Akk_nzpos[idx]] += kkt.jac[bm.jeq_Akk_coo[idx]]
        end

        # Scatter equality Jacobian coupling → C_dk
        @inbounds for idx in 1:length(bm.jeq_Cdk_coo)
            C_dk[bm.jeq_Cdk_row[idx], bm.jeq_Cdk_col[idx]] += kkt.jac[bm.jeq_Cdk_coo[idx]]
        end

        # Inequality condensation → A_kk (lower triangle)
        @inbounds for idx in 1:length(bm.ineq_Akk_nzpos)
            A_kk.nzval[bm.ineq_Akk_nzpos[idx]] += kkt.diag_buffer[bm.ineq_Akk_bufidx[idx]] *
                kkt.jac[bm.ineq_Akk_jcoo1[idx]] * kkt.jac[bm.ineq_Akk_jcoo2[idx]]
        end

        # Inequality condensation → C_dk
        @inbounds for idx in 1:length(bm.ineq_Cdk_row)
            C_dk[bm.ineq_Cdk_row[idx], bm.ineq_Cdk_col[idx]] += kkt.diag_buffer[bm.ineq_Cdk_bufidx[idx]] *
                kkt.jac[bm.ineq_Cdk_jcoo_d[idx]] * kkt.jac[bm.ineq_Cdk_jcoo_v[idx]]
        end

        # Factor A_kk
        factorize!(kkt.scenario_solvers[k])

        # Compute tmp = A_kk^{-1} * C_dk'  (blk × nd)
        buf = kkt.solve_buffers[k]
        for j in 1:nd
            @inbounds for i in 1:blk
                buf[i] = C_dk[j, i]  # C_dk' column j
            end
            solve_linear_system!(kkt.scenario_solvers[k], buf)
            @inbounds for i in 1:blk
                kkt.tmp_blk_nd[k][i, j] = buf[i]
            end
        end
    end

    # Phase 2 (sequential): accumulate into shared Schur complement S
    for k in 1:ns
        bm = kkt.block_maps[k]

        # Inequality condensation → S
        @inbounds for idx in 1:length(bm.ineq_S_row)
            S[bm.ineq_S_row[idx], bm.ineq_S_col[idx]] += kkt.diag_buffer[bm.ineq_S_bufidx[idx]] *
                kkt.jac[bm.ineq_S_jcoo1[idx]] * kkt.jac[bm.ineq_S_jcoo2[idx]]
        end

        # S -= C_dk * A_kk^{-1} * C_dk'
        mul!(S, kkt.C_dk[k], kkt.tmp_blk_nd[k], -one(T), one(T))
    end

    return
end

function factorize_kkt!(kkt::SchurComplementKKTSystem)
    return factorize!(kkt.linear_solver)
end

function solve_kkt!(
    kkt::SchurComplementKKTSystem,
    w::AbstractKKTVector{T},
) where T

    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    nc = kkt.nc
    n = num_variables(kkt)
    blk = kkt.blk_size

    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+kkt.n_ineq)
    wy = dual(w)

    Sigma_s = get_slack_regularization(kkt)

    reduce_rhs!(kkt, w)

    # Step 1: condense inequality contributions
    fill!(kkt.buffer, zero(T))
    if kkt.n_ineq > 0
        kkt.buffer[kkt.ind_ineq] .= kkt.diag_buffer .* (wy[kkt.ind_ineq] .+ ws ./ Sigma_s)
        # J' * buffer → wx (using sparse jt_csc)
        mul!(wx, kkt.jt_csc, kkt.buffer, one(T), one(T))
    end

    # Step 2: Extract per-scenario RHS blocks
    for k in 1:ns
        vr = (k-1)*nv+1 : k*nv
        rhs = kkt.rhs_k[k]
        @inbounds for i in 1:nv
            rhs[i] = wx[vr[1]+i-1]
        end
        for (ci, gi) in enumerate(kkt.eq_per_scenario[k])
            rhs[nv+ci] = wy[gi]
        end
    end
    @inbounds for i in 1:nd
        kkt.rhs_d[i] = wx[ns*nv+i]
    end

    # Step 3: Forward elimination
    # Phase 1 (parallel): solve per-scenario systems
    @blas_safe_threads for k in 1:ns
        solve_linear_system!(kkt.scenario_solvers[k], kkt.rhs_k[k])
    end
    # Phase 2 (sequential): accumulate into shared rhs_d
    for k in 1:ns
        mul!(kkt.rhs_d, kkt.C_dk[k], kkt.rhs_k[k], -one(T), one(T))
    end

    # Step 4: Solve Schur complement
    solve_linear_system!(kkt.linear_solver, kkt.rhs_d)

    # Step 5: Back-substitution (parallel — reads shared rhs_d, writes per-scenario rhs_k)
    @blas_safe_threads for k in 1:ns
        mul!(kkt.rhs_k[k], kkt.tmp_blk_nd[k], kkt.rhs_d, -one(T), one(T))
    end

    # Step 6: Write back to w
    for k in 1:ns
        vr = (k-1)*nv+1 : k*nv
        rhs = kkt.rhs_k[k]
        @inbounds for i in 1:nv
            wx[vr[1]+i-1] = rhs[i]
        end
        for (ci, gi) in enumerate(kkt.eq_per_scenario[k])
            wy[gi] = rhs[nv+ci]
        end
    end
    @inbounds for i in 1:nd
        wx[ns*nv+i] = kkt.rhs_d[i]
    end

    # Step 7: Recover inequality duals and slacks
    if kkt.n_ineq > 0
        eq_duals_backup = [wy[gi] for k in 1:ns for (_, gi) in enumerate(kkt.eq_per_scenario[k])]

        # J * Δx via sparse: (jt_csc)' * wx
        mul!(wy, kkt.jt_csc', wx)

        # Restore equality duals
        idx = 1
        for k in 1:ns
            for (_, gi) in enumerate(kkt.eq_per_scenario[k])
                wy[gi] = eq_duals_backup[idx]
                idx += 1
            end
        end

        # Inequality dual recovery
        @inbounds for idx in 1:length(kkt.ind_ineq)
            gi = kkt.ind_ineq[idx]
            wy[gi] = kkt.diag_buffer[idx] * wy[gi] - kkt.buffer[gi]
        end
        ws .= (ws .+ view(wy, kkt.ind_ineq)) ./ Sigma_s
    end

    finish_aug_solve!(kkt, w)
    return w
end

# KKT matrix-vector product for iterative refinement
function mul!(w::AbstractKKTVector{T}, kkt::SchurComplementKKTSystem{T}, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where T
    n = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    wx = @view(primal(w)[1:n])
    ws = @view(primal(w)[n+1:end])
    wy = dual(w)

    xx = @view(primal(x)[1:n])
    xs = @view(primal(x)[n+1:end])
    xy = dual(x)
    xz = @view(dual(x)[kkt.ind_ineq])

    # H * xx → wx (using sparse symmetric Hessian)
    wx .= beta .* wx
    mul!(wx, Symmetric(kkt.hess_csc, :L), xx, alpha, one(T))

    m = size(kkt.jt_csc, 2)
    if m > 0
        mul!(wx, kkt.jt_csc, dual(x), alpha, one(T))       # J' * xy
        mul!(wy, kkt.jt_csc', xx, alpha, beta)              # J * xx
    else
        wy .= beta .* wy
    end
    ws .= beta.*ws .- alpha.* xz
    @view(dual(w)[kkt.ind_ineq]) .-= alpha.* xs
    _kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function mul_hess_blk!(wx, kkt::SchurComplementKKTSystem, t)
    n = num_variables(kkt)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_csc, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
