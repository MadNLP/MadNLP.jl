
function factorize_wrapper!(ips::InteriorPointSolver)
    @trace(ips.logger,"Factorization started.")
    build_kkt!(ips.kkt)
    ips.cnt.linear_solver_time += @elapsed factorize!(ips.linear_solver)
end

function solve_refine_wrapper!(ips::InteriorPointSolver, x,b)
    cnt = ips.cnt
    @trace(ips.logger,"Iterative solution started.")
    fixed_variable_treatment_vec!(primaldual(b), ips.ind_fixed)

    cnt.linear_solver_time += @elapsed begin
        result = solve_refine!(primaldual(x), ips.iterator, primaldual(b))
    end

    if result == :Solved
        solve_status =  true
    else
        if improve!(ips.linear_solver)
            cnt.linear_solver_time += @elapsed begin
                factorize!(ips.linear_solver)
                ret = solve_refine!(primaldual(x), ips.iterator, primaldual(b))
                solve_status = (ret == :Solved)
            end
        else
            solve_status = false
        end
    end
    fixed_variable_treatment_vec!(primaldual(x), ips.ind_fixed)
    return solve_status
end

function solve_refine_wrapper!(ips::InteriorPointSolver{<:DenseCondensedKKTSystem}, x, b)
    cnt = ips.cnt
    @trace(ips.logger,"Iterative solution started.")
    fixed_variable_treatment_vec!(primaldual(b), ips.ind_fixed)

    kkt = ips.kkt

    n = num_variables(kkt)
    n_eq, ns = kkt.n_eq, kkt.n_ineq
    n_condensed = n + n_eq

    # load buffers
    b_c = view(ips._w1, 1:n_condensed)
    x_c = view(ips._w2, 1:n_condensed)
    jv_x = view(ips._w3, 1:ns) # for jprod
    jv_t = ips._w4x            # for jtprod
    v_c = ips._w4l

    Σs = get_slack_regularization(kkt)
    α = get_scaling_inequalities(kkt)

    # Decompose right hand side
    bx = view(b, 1:n)
    bs = view(b, n+1:n+ns)
    by = view(b, kkt.ind_eq_shifted)
    bz = view(b, kkt.ind_ineq_shifted)

    # Decompose results
    xx = view(x, 1:n)
    xs = view(x, n+1:n+ns)
    xy = view(x, kkt.ind_eq_shifted)
    xz = view(x, kkt.ind_ineq_shifted)

    v_c .= 0.0
    v_c[kkt.ind_ineq] .= (Σs .* bz .+ α .* bs) ./ α.^2
    jtprod!(jv_t, kkt, v_c)
    # init right-hand-side
    b_c[1:n] .= bx .+ jv_t[1:n]
    b_c[1+n:n+n_eq] .= by

    cnt.linear_solver_time += @elapsed (result = solve_refine!(x_c, ips.iterator, b_c))
    solve_status = (result == :Solved)

    # Expand solution
    xx .= x_c[1:n]
    xy .= x_c[1+n:end]
    jprod_ineq!(jv_x, kkt, xx)
    xz .= sqrt.(Σs) ./ α .* jv_x .- Σs .* bz ./ α.^2 .- bs ./ α
    xs .= (bs .+ α .* xz) ./ Σs

    fixed_variable_treatment_vec!(primaldual(x), ips.ind_fixed)
    return solve_status
end

