##################################################
##### GPU kernels for GPUSchurComplementKKT #####
##################################################

# Scatter Hessian diagonal entries → batched A_kk nzval
# Flattened: index i maps to entry i in a flat array of (coo, nzpos) pairs across all scenarios
@kernel function _scatter_hess_Akk_kernel!(
    nzval,
    @Const(hess),
    @Const(coo_flat),        # flat: scenario 1 entries, then scenario 2, etc.
    @Const(nzpos_flat),      # flat nzpos
    @Const(nnz_per_block),
    @Const(n_per_s),         # entries per scenario
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    idx = (i - 1) % n_per_s + 1
    flat = (k - 1) * n_per_s + idx
    offset = (k - 1) * nnz_per_block
    @inbounds nzval[offset + nzpos_flat[flat]] += hess[coo_flat[flat]]
end

# Scatter Hessian coupling entries → C_dk_batched (nd × blk_size × ns)
@kernel function _scatter_hess_Cdk_kernel!(
    C_dk,                    # (nd, blk_size, ns)
    @Const(hess),
    @Const(coo_flat),
    @Const(row_flat),        # design var index (1:nd)
    @Const(col_flat),        # scenario local var (1:nv)
    @Const(n_per_s),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    idx = (i - 1) % n_per_s + 1
    flat = (k - 1) * n_per_s + idx
    @inbounds C_dk[row_flat[flat], col_flat[flat], k] += hess[coo_flat[flat]]
end

# Scatter pr_diag or du_diag → batched A_kk diagonal
@kernel function _scatter_diag_Akk_kernel!(
    nzval,
    @Const(diag_vals),       # global pr_diag or du_diag
    @Const(global_flat),     # global index into diag_vals
    @Const(nzpos_flat),      # A_kk nzval position
    @Const(nnz_per_block),
    @Const(n_per_s),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    idx = (i - 1) % n_per_s + 1
    flat = (k - 1) * n_per_s + idx
    offset = (k - 1) * nnz_per_block
    @inbounds nzval[offset + nzpos_flat[flat]] += diag_vals[global_flat[flat]]
end

# Scatter equality Jacobian → A_kk lower triangle
@kernel function _scatter_jeq_Akk_kernel!(
    nzval,
    @Const(jac),
    @Const(coo_flat),
    @Const(nzpos_flat),
    @Const(nnz_per_block),
    @Const(n_per_s),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    idx = (i - 1) % n_per_s + 1
    flat = (k - 1) * n_per_s + idx
    offset = (k - 1) * nnz_per_block
    @inbounds nzval[offset + nzpos_flat[flat]] += jac[coo_flat[flat]]
end

# Scatter equality Jacobian coupling → C_dk_batched
@kernel function _scatter_jeq_Cdk_kernel!(
    C_dk,                    # (nd, blk_size, ns)
    @Const(jac),
    @Const(coo_flat),
    @Const(row_flat),
    @Const(col_flat),
    @Const(n_per_s),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    idx = (i - 1) % n_per_s + 1
    flat = (k - 1) * n_per_s + idx
    @inbounds C_dk[row_flat[flat], col_flat[flat], k] += jac[coo_flat[flat]]
end

# Inequality condensation → A_kk (lower triangle)
@kernel function _ineq_condense_Akk_kernel!(
    nzval,
    @Const(jac),
    @Const(diag_buffer),
    @Const(nzpos_flat),
    @Const(jcoo1_flat),
    @Const(jcoo2_flat),
    @Const(bufidx_flat),
    @Const(nnz_per_block),
    @Const(n_per_s),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    idx = (i - 1) % n_per_s + 1
    flat = (k - 1) * n_per_s + idx
    offset = (k - 1) * nnz_per_block
    @inbounds nzval[offset + nzpos_flat[flat]] += diag_buffer[bufidx_flat[flat]] *
        jac[jcoo1_flat[flat]] * jac[jcoo2_flat[flat]]
end

# Inequality condensation → C_dk
@kernel function _ineq_condense_Cdk_kernel!(
    C_dk,
    @Const(jac),
    @Const(diag_buffer),
    @Const(row_flat),
    @Const(col_flat),
    @Const(jcoo_d_flat),
    @Const(jcoo_v_flat),
    @Const(bufidx_flat),
    @Const(n_per_s),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    idx = (i - 1) % n_per_s + 1
    flat = (k - 1) * n_per_s + idx
    @inbounds C_dk[row_flat[flat], col_flat[flat], k] += diag_buffer[bufidx_flat[flat]] *
        jac[jcoo_d_flat[flat]] * jac[jcoo_v_flat[flat]]
end

# Inequality condensation → S (uses CUDA.@atomic for concurrent writes to small nd×nd matrix)
@kernel function _ineq_condense_S_kernel!(
    S,
    @Const(jac),
    @Const(diag_buffer),
    @Const(row_flat),
    @Const(col_flat),
    @Const(jcoo1_flat),
    @Const(jcoo2_flat),
    @Const(bufidx_flat),
    @Const(n_per_s),
    @Const(ns),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    idx = (i - 1) % n_per_s + 1
    flat = (k - 1) * n_per_s + idx
    if k <= ns
        @inbounds begin
            val = diag_buffer[bufidx_flat[flat]] *
                jac[jcoo1_flat[flat]] * jac[jcoo2_flat[flat]]
            CUDA.@atomic S[row_flat[flat], col_flat[flat]] += val
        end
    end
end

# Initialize S from design-design Hessian
@kernel function _init_S_hess_kernel!(
    S,
    @Const(hess),
    @Const(coo_indices),
    @Const(row_indices),
    @Const(col_indices),
)
    idx = @index(Global)
    @inbounds S[row_indices[idx], col_indices[idx]] += hess[coo_indices[idx]]
end

# Add pr_diag_dd to S diagonal
@kernel function _init_S_diag_kernel!(
    S,
    @Const(pr_diag),
    @Const(diag_offset),
    @Const(nd),
)
    idx = @index(Global)
    if idx <= nd
        @inbounds S[idx, idx] += pr_diag[diag_offset + idx]
    end
end

# Extract per-scenario RHS from global wx/wy → rhs_k_batched (blk_size × ns)
@kernel function _extract_rhs_kernel!(
    rhs_k,
    @Const(wx),
    @Const(wy),
    @Const(eq_global_indices),
    @Const(nv),
    @Const(nc_eq),
    @Const(ns),
    @Const(blk_size),
)
    i = @index(Global)
    k = (i - 1) ÷ blk_size + 1
    local_idx = (i - 1) % blk_size + 1
    if k <= ns
        if local_idx <= nv
            gi = (k - 1) * nv + local_idx
            @inbounds rhs_k[local_idx, k] = wx[gi]
        else
            eq_idx = local_idx - nv
            flat_idx = (k - 1) * nc_eq + eq_idx
            @inbounds rhs_k[local_idx, k] = wy[eq_global_indices[flat_idx]]
        end
    end
end

# Write back per-scenario solution to global wx/wy
@kernel function _writeback_rhs_kernel!(
    wx,
    wy,
    @Const(rhs_k),
    @Const(eq_global_indices),
    @Const(nv),
    @Const(nc_eq),
    @Const(ns),
    @Const(blk_size),
)
    i = @index(Global)
    k = (i - 1) ÷ blk_size + 1
    local_idx = (i - 1) % blk_size + 1
    if k <= ns
        if local_idx <= nv
            gi = (k - 1) * nv + local_idx
            @inbounds wx[gi] = rhs_k[local_idx, k]
        else
            eq_idx = local_idx - nv
            flat_idx = (k - 1) * nc_eq + eq_idx
            @inbounds wy[eq_global_indices[flat_idx]] = rhs_k[local_idx, k]
        end
    end
end
