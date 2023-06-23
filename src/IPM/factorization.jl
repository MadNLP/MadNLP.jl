
function factorize_wrapper!(solver::MadNLPSolver)
    @trace(solver.logger,"Factorization started.")
    build_kkt!(solver.kkt)
    solver.cnt.linear_solver_time += @elapsed factorize!(solver.linear_solver)
end

@inbounds function _kktmul!(w,x,del_w,du_diag,zl_r,zu_r,xl_r,xu_r,x_lr,x_ur)
    primal(w) .-= del_w .* primal(x)
    dual(w) .-= du_diag .* dual(x)
    w.xp_lr .+= dual_lb(x)
    w.xp_ur .-= dual_ub(x)            
    dual_lb(w) .+= .- x.xp_lr .* zl_r .+ dual_lb(x) .* (xl_r .- x_lr)
    dual_ub(w) .+= .- x.xp_ur .* zu_r .+ dual_ub(x) .* (xu_r .- x_ur)
end

@inbounds function aug_rhs_prep(
    xp_lr,wl,xl_r,x_lr,
    xp_ur,wu,x_ur,xu_r,
    )
    @simd for i=1:length(xp_lr)
        xp_lr[i] -= wl[i] / (xl_r[i] - x_lr[i])
    end
    @simd for i=1:length(xp_ur)
        xp_ur[i] += wu[i] / (xu_r[i] - x_ur[i])
    end
end
@inbounds function g(ws,wz,du_diag,jv_x,Σs) 
    @simd for i=1:length(ws)
        ws[i] = (- (wz[i] + du_diag[i] * ws[i]) + jv_x[i] ) / ( 1 - Σs[i] * du_diag[i] )
    end
end

@inbounds function solve_refine_wrapper!(
    solver::MadNLPSolver,
    x::AbstractKKTVector,
    b::AbstractKKTVector,
    w::AbstractKKTVector;
    resto = false,
    init = false,
    )
        cnt = solver.cnt
        @trace(solver.logger,"Iterative solution started.")

        kkt = solver.kkt

        fill!(full(x), 0)
        norm_b = norm(b, Inf)
        
        copyto!(full(w), full(b))

        status = false
        for kk=1:10
    @time begin
            err = norm(primal(w), Inf) / (one(eltype(primal(x))) + norm_b)
            
            if kk>=2 && err <= 1e-5
                status = true
                break
            end

            aug_rhs_prep(w.xp_lr, dual_lb(w), solver.xl_r, solver.x_lr,w.xp_ur, dual_ub(w), solver.xu_r, solver.x_ur)
            cnt.linear_solver_time += @elapsed solve!(solver.linear_solver, primal_dual(w))
            finish_aug_solve!(solver, solver.kkt, w, solver.mu)

            axpy!(1., full(w), full(x))
            copyto!(full(w), full(b))
            mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), -1., 1.)
            mul!(primal(w), kkt.jac_com', dual(x), -1., 1.)
            mul!(dual(w), kkt.jac_com,  primal(x), -1., 1.)
            _kktmul!(w,x,solver.del_w,kkt.du_diag,solver.zl_r,solver.zu_r,solver.xl_r,solver.xu_r,solver.x_lr,solver.x_ur)

    end
            init && break
        end


        if resto
            error()
            RR = solver.RR
            finish_aug_solve_RR!(RR.dpp,RR.dnn,RR.dzp,RR.dzn,solver.y,dual(solver.d),RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,solver.opt.rho)
        end

    return status
end

function solve_refine_wrapper!(
    solver::MadNLPSolver{T,<:DenseCondensedKKTSystem},
    x::AbstractKKTVector,
    b::AbstractKKTVector;
    resto = false
) where T
    cnt = solver.cnt
    @trace(solver.logger,"Iterative solution started.")

    kkt = solver.kkt

    n = num_variables(kkt)
    n_eq, ns = kkt.n_eq, kkt.n_ineq
    n_condensed = n + n_eq

    # load buffers
    b_c = view(full(solver._w1), 1:n_condensed)
    x_c = view(full(solver._w2), 1:n_condensed)
    jv_x = view(full(solver._w3), 1:ns) # for jprod
    jv_t = primal(solver._w4)             # for jtprod
    v_c = dual(solver._w4)

    Σs = get_slack_regularization(kkt)
    α = get_scaling_inequalities(kkt)

    # Decompose right hand side
    bx = view(full(b), 1:n)
    bs = view(full(b), n+1:n+ns)
    by = view(full(b), kkt.ind_eq_shifted)
    bz = view(full(b), kkt.ind_ineq_shifted)

    # Decompose results
    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+ns)
    xy = view(full(x), kkt.ind_eq_shifted)
    xz = view(full(x), kkt.ind_ineq_shifted)

    fill!(v_c, zero(T))
    v_c[kkt.ind_ineq] .= (Σs .* bz .+ α .* bs) ./ α.^2
    jtprod!(jv_t, kkt, v_c)
    # init right-hand-side
    b_c[1:n] .= bx .+ jv_t[1:n]
    b_c[1+n:n+n_eq] .= by

    cnt.linear_solver_time += @elapsed (result = solve_refine!(x_c, solver.iterator, b_c))
    solve_status = (result == :Solved)

    # Expand solution
    xx .= x_c[1:n]
    xy .= x_c[1+n:end]
    jprod_ineq!(jv_x, kkt, xx)
    xz .= sqrt.(Σs) ./ α .* jv_x .- Σs .* bz ./ α.^2 .- bs ./ α
    xs .= (bs .+ α .* xz) ./ Σs

    finish_aug_solve!(solver, solver.kkt, solver.mu)

    if resto
        RR = solver.RR
        finish_aug_solve_RR!(RR.dpp,RR.dnn,RR.dzp,RR.dzn,solver.y,dual(solver.d),RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,solver.opt.rho)
    end

    return solve_status
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

function solve_refine_wrapper!(
    solver::MadNLPSolver{T, <:SparseKKTSystem{T, VT, MT, QN}},
    x::AbstractKKTVector,
    b::AbstractKKTVector;
    resto = false
) where {T, VT, MT, QN<:CompactLBFGS{T, Vector{T}, Matrix{T}}}
    cnt = solver.cnt
    kkt = solver.kkt
    qn = kkt.quasi_newton
    n, p = size(qn)
    # Load buffers
    xr = qn._w2
    Tk = qn.Tk  ; fill!(Tk, zero(T))
    x_ = primal_dual(x)
    b_ = primal_dual(b)
    nn = length(x_)
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
    cnt.linear_solver_time += @elapsed begin
        result = solve_refine!(x, solver.iterator, b)
    end

    # Add low-rank correction
    if p > 0
        _init_lbfgs_factors!(qn.V1, qn.V2, qn.U, n, p)

        cnt.linear_solver_time += @elapsed begin
            multi_solve!(solver.linear_solver, qn.V2)  # V2 = C⁻¹ U
        end

        Tk[diagind(Tk)] .= one(T)                   # Tₖ = I
        mul!(Tk, qn.V1', qn.V2, one(T), one(T))     # Tₖ = (I + Vᵀ C⁻¹ U)
        J1 = qr(Tk)                                 # Tₖ⁻¹

        mul!(xr, qn.V1', x_)                        # xᵣ = Vᵀ C⁻¹ b
        ldiv!(J1, xr)                               # xᵣ = (I + Vᵀ C⁻¹ U)⁻¹ Vᵀ C⁻¹ b
        mul!(x_, qn.V2, xr, -one(T), one(T))        # x = x - C⁻¹ U xᵣ
    end

    finish_aug_solve!(solver, solver.kkt, solver.mu)
    if resto
        RR = solver.RR
        finish_aug_solve_RR!(RR.dpp,RR.dnn,RR.dzp,RR.dzn,solver.y,dual(solver.d),RR.pp,RR.nn,RR.zp,RR.zn,RR.mu_R,solver.opt.rho)
    end
    
    return result == :Solved
end


function solve_refine_wrapper!(
    solver::MadNLPSolver{T,<:SparseCondensedKKTSystem},
    x::AbstractKKTVector,
    b::AbstractKKTVector,
    w::AbstractKKTVector;
    resto = false,
    init = false,
    ) where T

    cnt = solver.cnt
    @trace(solver.logger,"Iterative solution started.")

    kkt = solver.kkt
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

   
    # load buffers
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

    # Decompose buffers
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2*m)
    
    fill!(full(x), 0)
    copyto!(full(w), full(b))
    
    norm_b = norm(b, Inf)
    for kk=1:10
        err = norm(full(w), Inf) / (one(eltype(full(x))) + norm_b)
        println(err)
        if err <= 1e-10
            break
        end
        aug_rhs_prep(
            w.xp_lr, dual_lb(w), solver.xl_r, solver.x_lr,
            w.xp_ur, dual_ub(w), solver.xu_r, solver.x_ur,
        )

        ##

        v_c .= kkt.diag_buffer .* (wz .+ ws ./ Σs) 
        
        # init right-hand-side
        mul!(jv_t, kkt.jt_csc, v_c)
        axpy!(1, jv_t, wx)

        cnt.linear_solver_time += @elapsed solve!(solver.linear_solver, wx)

        mul!(jv_x, kkt.jt_csc', wx)
        wz .= .- v_c .+ kkt.diag_buffer .* jv_x
        ws .= (ws .+ wz) ./ Σs

        ##
        # mul!(jv_x, kkt.jt_csc', wx)
        
        # copyto!(v_c, ws)
        # g(ws,wz,kkt.du_diag,jv_x,Σs) 
        # wz .= .-v_c .+ Σs .* ws

        ##
        finish_aug_solve!(solver, solver.kkt, w, solver.mu)
        
        axpy!(1., full(w), full(x))

        copyto!(full(w), full(b))
        
        mul!(wx, Symmetric(kkt.hess_com, :L), xx, -1., 1.)
        mul!(wx, kkt.jt_csc,  xz, -1., 1.)
        mul!(wz, kkt.jt_csc', xx, -1., 1.)
        axpy!(1, xz, ws)
        axpy!(1, xs, wz)
        
        _kktmul!(w,x,solver.del_w,kkt.du_diag,solver.zl_r,solver.zu_r,solver.xl_r,solver.xu_r,solver.x_lr,solver.x_ur)
        if init
            break
        end
        
    end

    return true
end
