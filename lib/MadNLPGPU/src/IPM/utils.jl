@inline function count_lu_bounds(lb::AbstractGPUVectorOrSubVector{T}, ub::AbstractGPUVectorOrSubVector{T}) where {T}
    num_lu_bounds = sum((g_lb .!= -Inf) .& (g_ub .!=  Inf))
    num_le_bounds = sum((g_lb .!= -Inf) .& (g_ub .==  Inf))
    num_ue_bounds = sum((g_ub .!=  Inf) .& (g_lb .== -Inf))
    return num_lu_bounds, num_le_bounds, num_ue_bounds
end
