function get_index_constraints(lvar, uvar, lcon, ucon, fixed_variable_treatment)
    ind_ineq = findall(lcon .!= ucon)
    xl = [lvar;view(lcon,ind_ineq)]
    xu = [uvar;view(ucon,ind_ineq)]
    if fixed_variable_treatment == MakeParameter
        ind_fixed = findall(xl .== xu)
        ind_lb = findall((xl .!= -Inf) .* (xl .!= xu))
        ind_ub = findall((xu .!=  Inf) .* (xl .!= xu))
    else
        ind_fixed = similar(xl, Int, 0)
        ind_lb = findall(xl .!=-Inf)
        ind_ub = findall(xu .!= Inf)
    end

    ind_llb = findall((lvar .!= -Inf).*(uvar .== Inf))
    ind_uub = findall((lvar .== -Inf).*(uvar .!= Inf))

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

