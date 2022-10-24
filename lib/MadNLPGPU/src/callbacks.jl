
function _init_buffer_bfgs!(kkt::MadNLP.AbstractKKTSystem{T, VT, MT, QN}, n, m) where {T, VT, MT, QN}
    haskey(kkt.etc, :x_gh) || (kkt.etc[:x_g] = zeros(T, n))
    haskey(kkt.etc, :j_gh) || (kkt.etc[:j_g] = zeros(T, n))
    haskey(kkt.etc, :j_gd) || (kkt.etc[:j_g] = VT(undef, n))
    return
end

function MadNLP.eval_lag_hess_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::MadNLP.AbstractKKTSystem{T, VT, MT, QN},
    x::Vector{T},
    l::Vector{T};
    is_resto=false,
) where {T, VT<:CuVector{T}, MT<:CuMatrix{T}, QN<:MadNLP.AbstractQuasiNewton{T, VT}}
    nlp = solver.nlp
    cnt = solver.cnt
    @trace(solver.logger, "Update BFGS matrices.")

    qn = kkt.qn
    Bk = kkt.hess
    sk, yk = qn.sk, qn.yk
    n = length(qn.sk)
    m = size(kkt.jac, 1)

    # Load the buffers to transfer data between the host and the device.
    _init_buffer_bfgs!(kkt, n, m)
    x_g = get(kkt.etc, :x_gh)
    j_g = get(kkt.etc, :j_gh) # on host
    j_d = get(kkt.etc, :j_gd) # on device
    # Init buffers.
    copyto!(x_g, qn.last_x)
    fill!(j_d, zero(T))
    fill!(j_g, zero(T))

    if cnt.obj_grad_cnt >= 2
        # Build sk = x+ - x
        copyto!(sk, 1, solver.x, 1, n)   # sₖ = x₊
        axpy!(-one(T), qn.last_x, sk)    # sₖ = x₊ - x

        # Build yk = ∇L+ - ∇L
        copyto!(yk, 1, solver.f, 1, n)   # yₖ = ∇f₊
        axpy!(-one(T), qn.last_g, yk)    # yₖ = ∇f₊ - ∇f
        if m > 0
            jtprod!(solver.jacl, kkt, l_g)
            copyto!(j_d, 1, solver.jacl, 1, n)
            yk .+= j_d                   # yₖ += J₊ᵀ l₊
            NLPModels.jtprod!(nlp, x_g, l, j_g)
            copyto!(qn.last_jv, j_g)
            axpy!(-one(T), qn.last_jv, yk)        # yₖ += J₊ᵀ l₊ - Jᵀ l₊
        end

        if cnt.obj_grad_cnt == 2
            init!(qn, Bk, sk, yk)
        end
        update!(qn, Bk, sk, yk)
    end

    # Backup data for next step
    copyto!(qn.last_x, 1, solver.x, 1, n)
    copyto!(qn.last_g, 1, solver.f, 1, n)

    compress_hessian!(kkt)
    return get_hessian(kkt)
end

