function get_index_constraints(nlp::AbstractNLPModel; fixed_variable_treatment=MAKE_PARAMETER)
    ind_ineq = findall(get_lcon(nlp) .!= get_ucon(nlp))
    xl = [get_lvar(nlp);view(get_lcon(nlp),ind_ineq)]
    xu = [get_uvar(nlp);view(get_ucon(nlp),ind_ineq)]
    if fixed_variable_treatment == MAKE_PARAMETER
        ind_fixed = findall(xl .== xu)
        ind_lb = findall((xl .!= -Inf) .* (xl .!= xu))
        ind_ub = findall((xu .!=  Inf) .* (xl .!= xu))
    else
        ind_fixed = similar(xl, Int, 0)
        ind_lb = findall(xl .!=-Inf)
        ind_ub = findall(xu .!= Inf)
    end

    ind_llb = findall((get_lvar(nlp) .!= -Inf).*(get_uvar(nlp) .== Inf))
    ind_uub = findall((get_lvar(nlp) .== -Inf).*(get_uvar(nlp) .!= Inf))

    # Return named tuple
    return (
        ind_ineq=ind_ineq,
        ind_fixed=ind_fixed,
        ind_lb=ind_lb,
        ind_ub=ind_ub,
        ind_llb=ind_llb,
        ind_uub=ind_uub,
    )
end

