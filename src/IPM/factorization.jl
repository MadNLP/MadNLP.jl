
function factorize_wrapper!(solver::MadNLPSolver)
    @trace(solver.logger,"Factorization started.")
    build_kkt!(solver.kkt)
    solver.cnt.linear_solver_time += @elapsed factorize!(solver.linear_solver)
end

function solve_refine_wrapper!(
    solver::MadNLPSolver,
    x::AbstractKKTVector,
    b::AbstractKKTVector,
)
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

function solve_refine_wrapper!(
    solver::MadNLPSolver{T,<:DenseCondensedKKTSystem},
    x::AbstractKKTVector,
    b::AbstractKKTVector,
) where T
    cnt = solver.cnt
    @trace(solver.logger,"Iterative solution started.")
    fixed_variable_treatment_vec!(full(b), solver.ind_fixed)

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

    v_c .= 0.0
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

    fixed_variable_treatment_vec!(full(x), solver.ind_fixed)
    return solve_status
end

