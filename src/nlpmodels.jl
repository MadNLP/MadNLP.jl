function get_index_constraints(lvar, uvar, lcon, ucon, fixed_variable_treatment, equality_treatment)
    
    ncon = length(lcon)
    
    if ncon > 0
        if equality_treatment == EnforceEquality
            ind_eq = findall(lcon .== ucon)
            ind_ineq = findall(lcon .!= ucon)
        else
            ind_eq = similar(lvar, Int, 0)
            ind_ineq = similar(lvar, Int, ncon) .= 1:ncon
        end
        xl = [lvar;view(lcon,ind_ineq)]
        xu = [uvar;view(ucon,ind_ineq)]
    else
        ind_eq   = similar(lvar, Int, 0)
        ind_ineq = similar(lvar, Int, 0)
        xl = lvar
        xu = uvar
    end
    
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
        ind_eq = ind_eq,
        ind_ineq = ind_ineq,
        ind_fixed = ind_fixed,
        ind_lb = ind_lb,
        ind_ub = ind_ub,
        ind_llb = ind_llb,
        ind_uub = ind_uub,
    )
end

