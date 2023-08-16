function solve_refine_wrapper!(d, solver, p, w)
    result = false
    
    solver.cnt.linear_solver_time += @elapsed begin
        if solve_refine!(d, solver.iterator, p, w)
            result = true
        else
            if improve!(solver.kkt.linear_solver)
                if solve_refine!(d, solver.iterator, p, w)
                    result = true
                end
            end
        end
    end

    return result
end
function factorize_wrapper!(solver::MadNLPSolver)
    @trace(solver.logger,"Factorization started.")
    build_kkt!(solver.kkt)
    solver.cnt.linear_solver_time += @elapsed factorize!(solver.kkt.linear_solver)
end

function solve!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    wzl = dual_lb(w) 
    wzu = dual_ub(w)
    f(x,y) = iszero(y) ? x : x/y
    wzl .= f.(wzl, kkt.l_lower_aug)
    wzu .= f.(wzu, kkt.u_lower_aug)
    solve!(kkt.linear_solver, full(w))
    wzl .*= .-kkt.l_lower_aug
    wzu .*= kkt.u_lower_aug
end

function solve!(kkt::AbstractReducedKKTSystem, w::AbstractKKTVector)
    reduce_rhs!(w.xp_lr, dual_lb(w), kkt.l_diag, w.xp_ur, dual_ub(w), kkt.u_diag)
    solve!(kkt.linear_solver, primal_dual(w))
    finish_aug_solve!(kkt, w)
end

function solve!(
    kkt::SparseKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
        
    qn = kkt.quasi_newton
    n, p = size(qn)
    # Load buffers
    xr = qn._w2
    Tk = qn.Tk
    w_ = primal_dual(w)
    nn = length(w_)

    fill!(Tk, zero(T))
    reduce_rhs!(w.xp_lr, dual_lb(w), kkt.l_diag, w.xp_ur, dual_ub(w), kkt.u_diag)


    # Resize arrays with correct dimension
    if size(qn.V1, 2) < 2*p
        qn.V1 = zeros(nn, 2*p)
        qn.V2 = zeros(nn, 2*p)
    else
        fill!(qn.V1, zero(T))
        fill!(qn.V2, zero(T))
    end

    # Solve LBFGS system with Sherman-Morrison-Woodbury formula
    # (C + U Vᵀ)⁻¹ = C⁻¹ - C⁻¹ U (I + Vᵀ C⁻¹ U) Vᵀ C⁻¹

    # Solve linear system without low-rank part
    solve!(kkt.linear_solver, primal_dual(w))

    # Add low-rank correction
    if p > 0
        _init_lbfgs_factors!(qn.V1, qn.V2, qn.U, n, p)

        multi_solve!(kkt.linear_solver, qn.V2)  # V2 = C⁻¹ U

        Tk[diagind(Tk)] .= one(T)                   # Tₖ = I
        mul!(Tk, qn.V1', qn.V2, one(T), one(T))     # Tₖ = (I + Vᵀ C⁻¹ U)
        J1 = qr(Tk)                                 # Tₖ⁻¹

        mul!(xr, qn.V1', w_)                        # xᵣ = Vᵀ C⁻¹ b
        ldiv!(J1, xr)                               # xᵣ = (I + Vᵀ C⁻¹ U)⁻¹ Vᵀ C⁻¹ b
        mul!(w_, qn.V2, xr, -one(T), one(T))        # x = x - C⁻¹ U xᵣ
    end

    finish_aug_solve!(kkt, w)    
end


function solve!(kkt::SparseCondensedKKTSystem{T}, w::AbstractKKTVector)  where T

    (n,m) = size(kkt.jt_csc)

    # Decompose buffers
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2*m)
    Σs = view(kkt.pr_diag, n+1:n+m)
    
    reduce_rhs!(w.xp_lr, dual_lb(w), kkt.l_diag, w.xp_ur, dual_ub(w), kkt.u_diag)

    kkt.buffer .= kkt.diag_buffer .* (wz .+ ws ./ Σs) 
    
    mul!(wx, kkt.jt_csc, kkt.buffer, one(T), one(T))
    solve!(kkt.linear_solver, wx)

    mul!(kkt.buffer2, kkt.jt_csc', wx) # TODO: investigate why directly using wz here is causing an error

    wz .= .- kkt.buffer .+ kkt.diag_buffer .* kkt.buffer2
    ws .= (ws .+ wz) ./ Σs

    finish_aug_solve!(kkt, w)

end

function solve!( 
    kkt::DenseCondensedKKTSystem,
    w::AbstractKKTVector{T},
    ) where T

    n = num_variables(kkt)
    n_eq, ns = kkt.n_eq, kkt.n_ineq
    n_condensed = n + n_eq
    
    # Decompose rhs
    wx = view(full(w), 1:n)
    ws = view(full(w), n+1:n+ns)
    wy = view(full(w), kkt.ind_eq_shifted)
    wz = view(full(w), kkt.ind_ineq_shifted)

    x = kkt.pd_buffer
    xx = view(x, 1:n)
    xy = view(x, n+1:n+n_eq)

    Σs = get_slack_regularization(kkt)

    reduce_rhs!(w.xp_lr, dual_lb(w), kkt.l_diag, w.xp_ur, dual_ub(w), kkt.u_diag)

    fill!(kkt.buffer, zero(T))
    kkt.buffer[kkt.ind_ineq] .= kkt.diag_buffer .* (wz .+ ws ./ Σs) 
    mul!(xx, kkt.jac', kkt.buffer)
    xx .+= wx
    xy .= wy
    solve!(kkt.linear_solver, x)

    wx .= xx
    mul!(dual(w), kkt.jac, wx)
    wy .= xy
    wz .*= kkt.diag_buffer
    dual(w) .-= kkt.buffer
    ws .= (ws .+ wz) ./ Σs
    
    finish_aug_solve!(kkt, w)
end

function mul!(w::AbstractKKTVector{T}, kkt::Union{SparseKKTSystem,SparseUnreducedKKTSystem}, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where T
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    _kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
end

function mul!(w::AbstractKKTVector{T}, kkt::SparseCondensedKKTSystem, x::AbstractKKTVector, alpha, beta) where T
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    # Decompose results
    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+m)
    xz = view(full(x), n+m+1:n+2*m)

    # Decompose buffers
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2*m)

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta) # TODO: make this symmetric

    mul!(wx, kkt.jt_csc,  xz, alpha, beta)
    mul!(wz, kkt.jt_csc', xx, alpha, one(T))
    axpy!(-alpha, xz, ws)
    axpy!(-alpha, xs, wz)
    
    _kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
end

function mul!(w::AbstractKKTVector{T}, kkt::AbstractDenseKKTSystem, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where T
    (m, n) = size(kkt.jac)
    wx = @view(primal(w)[1:n])
    ws = @view(primal(w)[n+1:end])
    wy = dual(w)
    wz = @view(dual(w)[kkt.ind_ineq])
    
    xx = @view(primal(x)[1:n])
    xs = @view(primal(x)[n+1:end])
    xy = dual(x)
    xz = @view(dual(x)[kkt.ind_ineq])
    
    symul!(wx, kkt.hess, xx, alpha, beta)
    if m > 0  # otherwise, CUDA causes an error
        mul!(wx, kkt.jac', dual(x), alpha, one(T))
        mul!(wy, kkt.jac,  xx, alpha, beta)
    end
    ws .= beta.*ws .- alpha.* xz
    wz .= beta.*wz .- alpha.* xs
    _kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
end

function mul_hess_blk!(wx, kkt::Union{DenseKKTSystem,DenseCondensedKKTSystem}, t)
    n = size(kkt.hess, 1)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
    
function mul_hess_blk!(wx, kkt::Union{SparseKKTSystem,SparseCondensedKKTSystem}, t)
    n = size(kkt.hess_com, 1)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_com, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
function mul_hess_blk!(wx, kkt::SparseUnreducedKKTSystem, t)
    ind_lb = kkt.ind_lb
    ind_ub = kkt.ind_ub
    
    n = size(kkt.hess_com, 1)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_com, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
    wx[ind_lb] .-= @view(t[ind_lb]) .* (kkt.l_lower ./ kkt.l_diag)
    wx[ind_ub] .-= @view(t[ind_ub]) .* (kkt.u_lower ./ kkt.u_diag)
end

# Set V1 = [U₁   U₂]   ,   V2 = [-U₁   U₂]
function _init_lbfgs_factors!(V1, V2, U, n, p)
    @inbounds for i in 1:n, j in 1:p
        V1[i, j] = U[i, j]
        V2[i, j] = -U[i, j]
        V1[i, j+p] = U[i, j+p]
        V2[i, j+p] = U[i, j+p]
    end
end

