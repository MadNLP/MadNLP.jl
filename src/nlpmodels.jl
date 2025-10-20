
"""
    AbstractFixedVariableTreatment

Abstract type to define the reformulation of the fixed variables inside MadNLP.
"""
abstract type AbstractFixedVariableTreatment end

"""
    MakeParameter{VT, VI} <: AbstractFixedVariableTreatment

Remove the fixed variables from the optimization variables and
define them as problem's parameters.
"""
struct MakeParameter{VT,VI} <: AbstractFixedVariableTreatment
    fixed::VI
    fixedj::VI
    fixedh::VI
    grad_storage::VT
end

"""
    RelaxBound <: AbstractFixedVariableTreatment

Relax the fixed variables ``x = x_{fixed}`` as bounded
variables ``x_{fixed} - ϵ ≤ x ≤ x_{fixed} + ϵ``, with
``ϵ`` a small-enough parameter.
"""
struct RelaxBound <: AbstractFixedVariableTreatment end


"""
    AbstractEqualityTreatment

Abstract type to define the reformulation of the equality
constraints inside MadNLP.
"""
abstract type AbstractEqualityTreatment end

"""
    EnforceEquality <: AbstractEqualityTreatment

Keep the equality constraints intact.

The solution returned by MadNLP will respect the equality constraints.
"""
struct EnforceEquality <: AbstractEqualityTreatment end

"""
    RelaxEquality <: AbstractEqualityTreatment

Relax the equality constraints ``g(x) = 0`` with two
inequality constraints, as ``-ϵ ≤ g(x) ≤ ϵ``. The parameter
``ϵ`` is usually small.

The solution returned by MadNLP will satisfy the equality
constraints only up to a tolerance ``ϵ``.

"""
struct RelaxEquality <: AbstractEqualityTreatment end


"""
    get_index_constraints(nlp::AbstractNLPModel)

Analyze the bounds of the variables and the constraints in the `AbstractNLPModel` `nlp`.
Return a named-tuple witht the following keys:return (

* `ind_eq`: indices of equality constraints.
* `ind_ineq`: indices of inequality constraints.
* `ind_fixed`: indices of fixed variables.
* `ind_lb`: indices of variables with a lower-bound.
* `ind_ub`: indices of variables with an upper-bound.
* `ind_llb`: indices of variables with *only* a lower-bound.
* `ind_uub`: indices of variables with *only* an upper-bound.

"""
function get_index_constraints(
    @nospecialize(nlp::AbstractNLPModel); options...
)
    get_index_constraints(
        get_lvar(nlp), get_uvar(nlp),
        get_lcon(nlp), get_ucon(nlp);
        options...
    )
end

function get_index_constraints(
    lvar, uvar,
    lcon, ucon;
    fixed_variable_treatment=MakeParameter,
    equality_treatment=EnforceEquality,
)
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

"""
    AbstractCallback{T, VT}

Wrap the `AbstractNLPModel` passed by the user in a form amenable to MadNLP.

An `AbstractCallback` handles the scaling of the problem and the
reformulations of the equality constraints and fixed variables.

"""
abstract type AbstractCallback{T,VT} end

"""
    create_callback(
        ::Type{Callback},
        nlp::AbstractNLPModel{T, VT};
        fixed_variable_treatment=MakeParameter,
        equality_treatment=EnforceEquality,
    ) where {T, VT}

Wrap the nonlinear program `nlp` using the callback wrapper
with type `Callback`. The option `fixed_variable_treatment`
decides if the fixed variables are relaxed (`RelaxBound`)
or removed (`MakeParameter`). The option `equality_treatment`
decides if the the equality constraints are keep as is
(`EnforceEquality`) or relaxed (`RelaxEquality`).

"""
function create_callback end

"""
    SparseCallback{T, VT} < AbstractCallback{T, VT}

Wrap an `AbstractNLPModel` using sparse structures.

"""
struct SparseCallback{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    FH <: AbstractFixedVariableTreatment,
    EH <: AbstractEqualityTreatment,
    } <: AbstractCallback{T, VT}

    nlp::AbstractNLPModel{T, VT}
    nvar::Int
    ncon::Int
    nnzj::Int
    nnzh::Int

    con_buffer::VT
    jac_buffer::VT
    grad_buffer::VT
    hess_buffer::VT

    jac_I::VI
    jac_J::VI
    hess_I::VI
    hess_J::VI

    obj_scale::Base.RefValue{T}
    con_scale::VT
    jac_scale::VT

    fixed_handler::FH
    equality_handler::EH

    function SparseCallback(
        @nospecialize(nlp::AbstractNLPModel),
        n,m,nnzj,nnzh,
        con_buffer,
        jac_buffer,
        grad_buffer,
        hess_buffer,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        obj_scale,
        con_scale,
        jac_scale,
        fixed_handler,
        equality_handler
        )

        new{
            eltype(get_x0(nlp)),
            typeof(get_x0(nlp)),
            typeof(jac_I),
            typeof(fixed_handler),
            typeof(equality_handler)
        }(
            nlp,
            n,m,nnzj,nnzh,
            con_buffer,
            jac_buffer,
            grad_buffer,
            hess_buffer,
            jac_I,
            jac_J,
            hess_I,
            hess_J,
            obj_scale,
            con_scale,
            jac_scale,
            fixed_handler,
            equality_handler
        )
    end
end

"""
    DenseCallback{T, VT} < AbstractCallback{T, VT}

Wrap an `AbstractNLPModel` using dense structures.

"""
struct DenseCallback{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    FH <: AbstractFixedVariableTreatment,
    EH <: AbstractEqualityTreatment,
    } <: AbstractCallback{T, VT}

    nlp::AbstractNLPModel{T, VT}
    nvar::Int
    ncon::Int

    con_buffer::VT
    jac_buffer::MT
    grad_buffer::VT

    obj_scale::Base.RefValue{T}
    con_scale::VT

    fixed_handler::FH
    equality_handler::EH
end


create_array(cb::AbstractCallback, args...) = similar(get_x0(cb.nlp), args...)

function set_obj_scale!(obj_scale, f::VT, max_gradient) where {T, VT <: AbstractVector{T}}
    obj_scale[] = min(one(T), max_gradient / norm(f, Inf))
end

function set_con_scale_sparse!(con_scale::VT, jac_I,jac_buffer, max_gradient) where {T, VT <: AbstractVector{T}}
    fill!(con_scale, one(T))
    _set_con_scale_sparse!(con_scale, jac_I, jac_buffer)
    map!(x-> min(one(T), max_gradient / x), con_scale, con_scale)
end
function _set_con_scale_sparse!(con_scale, jac_I, jac_buffer)
    @inbounds @simd for i in 1:length(jac_I)
        row = jac_I[i]
        con_scale[row] = max(con_scale[row], abs(jac_buffer[i]))
    end
end

function set_jac_scale_sparse!(jac_scale::VT, con_scale, jac_I) where {T, VT <: AbstractVector{T}}
    copyto!(jac_scale,  @view(con_scale[jac_I]))
end

function set_con_scale_dense!(con_scale::VT, jac_buffer, max_gradient) where {T, VT <: AbstractVector{T}}
    con_scale .= min.(one(T), max_gradient ./ mapreduce(abs, max, jac_buffer, dims=2, init=one(T)))
end


function create_dense_fixed_handler(
    fixed_variable_treatment::Type{MakeParameter},
    @nospecialize(nlp::AbstractNLPModel),
)
    lvar = get_lvar(nlp)
    uvar = get_uvar(nlp)
    isfixed  = (lvar .== uvar)
    fixed  = findall(isfixed)
    return MakeParameter(
        fixed,
        similar(fixed,0),
        similar(fixed,0),
        similar(lvar, length(fixed))
    )
end

function create_sparse_fixed_handler(
    fixed_variable_treatment::Type{MakeParameter},
    @nospecialize(nlp::AbstractNLPModel),
    jac_I,
    jac_J,
    hess_I,
    hess_J,
    hess_buffer,
)
    lvar = get_lvar(nlp)
    uvar = get_uvar(nlp)
    nnzj = get_nnzj(nlp.meta)
    nnzh = get_nnzh(nlp.meta)

    isfixed  = (lvar .== uvar)

    fixed  = findall(isfixed)
    fixedj = findall(@view(isfixed[jac_J]))
    fixedh = findall(@view(isfixed[hess_I]) .|| @view(isfixed[hess_J]))
    nfixed = length(fixed)

    nnzh = nnzh + nfixed
    resize!(hess_I, nnzh)
    resize!(hess_J, nnzh)
    resize!(hess_buffer, nnzh)
    copyto!(@view(hess_I[end-nfixed+1:end]), fixed)
    copyto!(@view(hess_J[end-nfixed+1:end]), fixed)

    fixed_handler = MakeParameter(
        fixed,
        fixedj,
        fixedh,
        similar(lvar, length(fixed))
    )

    return fixed_handler, nnzj, nnzh
end

function create_sparse_fixed_handler(
    fixed_variable_treatment::Type{RelaxBound},
    @nospecialize(nlp::AbstractNLPModel),
    jac_I,
    jac_J,
    hess_I,
    hess_J,
    hess_buffer,
)
    fixed_handler = RelaxBound()
    return fixed_handler, get_nnzj(nlp.meta), get_nnzh(nlp.meta)
end

function create_callback(
    ::Type{SparseCallback},
    @nospecialize(nlp::AbstractNLPModel);
    fixed_variable_treatment=MakeParameter,
    equality_treatment=EnforceEquality,
    ) 

    x0   = get_x0(nlp)
    T = eltype(x0)
    VT = typeof(x0)

    n = get_nvar(nlp)
    m = get_ncon(nlp)
    nnzj = get_nnzj(nlp.meta)
    nnzh = get_nnzh(nlp.meta)
    
    jac_I = similar(x0, Int, nnzj)
    jac_J = similar(x0, Int, nnzj)
    hess_I = similar(x0, Int, nnzh)
    hess_J = similar(x0, Int, nnzh)

    con_buffer = similar(x0, m)     ; fill!(con_buffer, zero(T))
    grad_buffer = similar(x0, n)    ; fill!(grad_buffer, zero(T))
    jac_buffer = similar(x0, nnzj)  ; fill!(jac_buffer, zero(T))
    hess_buffer = similar(x0, nnzh) ; fill!(hess_buffer, zero(T))

    obj_scale = Ref(one(T))
    con_scale = similar(jac_buffer, m)    ; fill!(con_scale, one(T))
    jac_scale = similar(jac_buffer, nnzj) ; fill!(jac_scale, one(T))
    
    if nnzj > 0
        jac_structure!(nlp, jac_I, jac_J)
    end
    if nnzh > 0
        hess_structure!(nlp, hess_I, hess_J)
    end

    fixed_handler, nnzj, nnzh = create_sparse_fixed_handler(
        fixed_variable_treatment,
        nlp,
        jac_I, jac_J, hess_I, hess_J,
        hess_buffer,
    )
    equality_handler = equality_treatment()
    
    return SparseCallback(
        nlp,
        n,m,nnzj,nnzh,
        con_buffer,
        jac_buffer,
        grad_buffer,
        hess_buffer,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        obj_scale,
        con_scale,
        jac_scale,
        fixed_handler,
        equality_handler
    )
end

function create_callback(
    ::Type{DenseCallback},
    @nospecialize(nlp::AbstractNLPModel);
    fixed_variable_treatment=MakeParameter,
    equality_treatment=EnforceEquality,
    ) 

    n = get_nvar(nlp)
    m = get_ncon(nlp)

    x0   = similar(get_x0(nlp))
    T = eltype(x0)
    
    con_buffer = similar(x0, m) ;    fill!(con_buffer, zero(T))
    jac_buffer = similar(x0, m, n) ; fill!(jac_buffer, zero(T))
    grad_buffer = similar(x0, n) ;   fill!(grad_buffer, zero(T))
    obj_scale = Ref(one(T))
    con_scale = similar(x0, m) ; fill!(con_scale, one(T))

    fixed_handler = create_dense_fixed_handler(
        fixed_variable_treatment,
        nlp,
    )
    equality_handler = equality_treatment()

    return DenseCallback(
        nlp,
        n, m,
        con_buffer,
        jac_buffer,
        grad_buffer,
        obj_scale,
        con_scale,
        fixed_handler,
        equality_handler
    )
end

function _treat_fixed_variable_initialize!(fixed_handler::RelaxBound, x0, lvar, uvar) end
function _treat_fixed_variable_initialize!(fixed_handler::MakeParameter, x0, lvar, uvar)
    fixed = fixed_handler.fixed
    copyto!(@view(x0[fixed]), @view(lvar[fixed]))
    fill!(@view(lvar[fixed]), -Inf)
    fill!(@view(uvar[fixed]),  Inf)
end

function _treat_equality_initialize!(equality_handler::EnforceEquality, lcon, ucon, tol) end
function _treat_equality_initialize!(equality_handler::RelaxEquality, lcon, ucon, tol)
    set_initial_bounds!(
        lcon,
        ucon,
        tol
    )
end

function initialize!(
    cb::AbstractCallback,
    x, xl, xu, y0, rhs,
    ind_ineq;
    tol=1e-8,
    bound_push=1e-2,
    bound_fac=1e-2,
    )

    x0= variable(x)
    lvar= variable(xl)
    uvar= variable(xu)

    nlp = cb.nlp
    con_buffer =cb.con_buffer
    grad_buffer =cb.grad_buffer


    x0   .= get_x0(nlp)
    y0   .= get_y0(nlp)
    lvar .= get_lvar(nlp)
    uvar .= get_uvar(nlp)
    lcon = copy(get_lcon(nlp))
    ucon = copy(get_ucon(nlp))

    _treat_fixed_variable_initialize!(cb.fixed_handler, x0, lvar, uvar)
    _treat_equality_initialize!(cb.equality_handler, lcon, ucon, tol)

    set_initial_bounds!(
        lvar,
        uvar,
        tol
    )
    initialize_variables!(
        x0,
        lvar,
        uvar,
        bound_push,
        bound_fac
    )

    cons!(nlp,x0,con_buffer)

    slack(xl) .= view(lcon, ind_ineq)
    slack(xu) .= view(ucon, ind_ineq)
    rhs .= (lcon.==ucon).*lcon
    copyto!(slack(x), @view(con_buffer[ind_ineq]))

    set_initial_bounds!(
        slack(xl),
        slack(xu),
        tol
    )
    initialize_variables!(
        slack(x),
        slack(xl),
        slack(xu),
        bound_push,
        bound_fac
    )
end

function set_scaling!(
    cb::SparseCallback,
    x, xl, xu, y0, rhs,
    ind_ineq,
    nlp_scaling_max_gradient
    )
    @nospecialize
    x0= variable(x)

    nlp = cb.nlp
    obj_scale = cb.obj_scale
    con_scale = cb.con_scale
    jac_scale = cb.jac_scale
    con_buffer =cb.con_buffer
    jac_buffer =cb.jac_buffer
    grad_buffer =cb.grad_buffer

    # Set scaling
    jac_coord!(nlp,x0,jac_buffer)
    set_con_scale_sparse!(con_scale, cb.jac_I, jac_buffer, nlp_scaling_max_gradient)
    set_jac_scale_sparse!(jac_scale, con_scale, cb.jac_I)

    grad!(nlp,x0,grad_buffer)
    set_obj_scale!(obj_scale, grad_buffer, nlp_scaling_max_gradient)

    con_scale_slk = @view(con_scale[ind_ineq])
    y0  ./= con_scale
    rhs .*= con_scale
    slack(x) .*= con_scale_slk
    slack(xl) .*= con_scale_slk
    slack(xu) .*= con_scale_slk
    return
end

function set_scaling!(
    cb::DenseCallback,
    x, xl, xu, y0, rhs,
    ind_ineq,
    nlp_scaling_max_gradient
    )
    @nospecialize
    x0 = variable(x)

    nlp = cb.nlp
    obj_scale = cb.obj_scale
    con_scale = cb.con_scale
    con_buffer =cb.con_buffer
    jac_buffer =cb.jac_buffer
    grad_buffer =cb.grad_buffer

    # Set scaling
    jac_dense!(nlp,x0,jac_buffer)
    set_con_scale_dense!(con_scale, jac_buffer, nlp_scaling_max_gradient)

    grad!(nlp,x0,grad_buffer)
    set_obj_scale!(obj_scale, grad_buffer, nlp_scaling_max_gradient)

    con_scale_slk = @view(con_scale[ind_ineq])
    y0  ./= con_scale
    rhs .*= con_scale
    slack(x) .*= con_scale_slk
    slack(xl) .*= con_scale_slk
    slack(xu) .*= con_scale_slk
    return
end

function _jac_sparsity_wrapper!(
    cb::SparseCallback,
    I::AbstractVector,J::AbstractVector
    )

    copyto!(I, cb.jac_I)
    copyto!(J, cb.jac_J)
    return
end

function _hess_sparsity_wrapper!(
    cb::SparseCallback,
    I::AbstractVector,J::AbstractVector
    )
    copyto!(I, cb.hess_I)
    copyto!(J, cb.hess_J)
    return
end


function _eval_cons_wrapper!(
    cb::AbstractCallback,
    x::AbstractVector,
    c::AbstractVector
    )
    cons!(cb.nlp, x, c)
    c .*= cb.con_scale
    return c
end

function _eval_jac_wrapper!(
    cb::SparseCallback,
    x::AbstractVector,
    jac::AbstractVector
    )
    jac_coord!(cb.nlp, x, jac)
    jac .*= cb.jac_scale

    _treat_fixed_variable_jac_coord!(cb.fixed_handler, cb, x, jac)
end

function _eval_jtprod_wrapper!(
    cb::AbstractCallback{T},
    x::AbstractVector,
    v::AbstractVector,
    jvt::AbstractVector,
    ) where T

    y = cb.con_buffer
    y .= v .* cb.con_scale
    jtprod!(cb.nlp, x, y, jvt)
    _treat_fixed_variable_grad!(cb.fixed_handler, cb, x, jvt)
    return jvt
end

function _treat_fixed_variable_jac_coord!(fixed_handler::RelaxBound, cb, x, jac) end
function _treat_fixed_variable_jac_coord!(fixed_handler::MakeParameter, cb::SparseCallback{T}, x, jac) where T
    fill!(@view(jac[fixed_handler.fixedj]), zero(T))
end

function _eval_grad_f_wrapper!(
    cb::AbstractCallback,
    x::AbstractVector,
    grad::AbstractVector{T}
    )  where T
    grad!(cb.nlp, x, grad)
    grad .*= cb.obj_scale[]
    _treat_fixed_variable_grad!(cb.fixed_handler, cb, x, grad)
end
function _treat_fixed_variable_grad!(fixed_handler::RelaxBound, cb, x, grad) end
function _treat_fixed_variable_grad!(fixed_handler::MakeParameter, cb::AbstractCallback{T,VT}, x, grad) where {T,VT}
    lvar = get_lvar(cb.nlp)::VT
    fixed_handler.grad_storage .= @view(grad[fixed_handler.fixed])
    map!(
        (x,y)->x-y,
        @view(grad[fixed_handler.fixed]),
        @view(x[cb.fixed_handler.fixed]),
        @view(lvar[cb.fixed_handler.fixed])
    )
end

function _eval_f_wrapper(cb::AbstractCallback,x::AbstractVector{T}) where T
    return (obj(cb.nlp,x)::T) * cb.obj_scale[]
end

function _eval_lag_hess_wrapper!(
    cb::SparseCallback,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector{T};
    obj_weight::T = one(T)
    ) where T
    nnzh_orig = get_nnzh(cb.nlp)::Int

    cb.con_buffer .= y .* cb.con_scale
    hess_coord!(
        cb.nlp, x, cb.con_buffer, hess;
        obj_weight = obj_weight * cb.obj_scale[]
    )
    _treat_fixed_variable_hess_coord!(cb.fixed_handler, cb, hess)
    return
end

function _treat_fixed_variable_hess_coord!(fixed_handler::RelaxBound, cb, hess) end
function _treat_fixed_variable_hess_coord!(fixed_handler::MakeParameter, cb::SparseCallback{T}, hess::AbstractVector{T}) where T
    nnzh_orig = get_nnzh(cb.nlp)::Int
    fill!(@view(hess[fixed_handler.fixedh]), zero(T))
    fill!(@view(hess[nnzh_orig+1:end]), one(T))
end

function _eval_jac_wrapper!(
    cb::SparseCallback{T},
    x::AbstractVector,
    jac::AbstractMatrix
    ) where T

    jac_buffer = cb.jac_buffer
    _eval_jac_wrapper!(cb, x, jac_buffer)
    fill!(jac, zero(T))
    @inbounds @simd for k=1:length(cb.jac_I)
        i,j = cb.jac_I[k], cb.jac_J[k]
        jac[i,j] += jac_buffer[k]
    end
end

function _eval_lag_hess_wrapper!(
    cb::SparseCallback,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractMatrix{T};
    obj_weight = one(T)
    ) where {T}

    hess_buffer = cb.hess_buffer
    _eval_lag_hess_wrapper!(cb, x, y, hess_buffer; obj_weight=obj_weight * cb.obj_scale[])
    fill!(hess, zero(T))
    @inbounds @simd for k=1:length(cb.hess_I)
        i,j = cb.hess_I[k], cb.hess_J[k]
        hess[i,j] += hess_buffer[k]
    end
    _treat_fixed_variable_hess_dense!(cb.fixed_handler, cb, hess)
end
function _treat_fixed_variable_hess_dense!(fixed_handler::RelaxBound, cb, hess) end
function _treat_fixed_variable_hess_dense!(fixed_handler::MakeParameter, cb::SparseCallback{T}, hess) where T
    nnzh_orig = get_nnzh(cb.nlp)

    fixed = fixed_handler.fixed
    _set_diag!(hess, fixed, one(T))
end

function _eval_jac_wrapper!(
    cb::DenseCallback{T},
    x::AbstractVector,
    jac::AbstractMatrix
    ) where T

    jac_dense!(cb.nlp, x, jac)
    jac .*= cb.con_scale
    _treat_fixed_variable_jac_dense!(cb.fixed_handler, cb, jac)
end
function _treat_fixed_variable_jac_dense!(fixed_handler::RelaxBound, cb::DenseCallback, jac) end
function _treat_fixed_variable_jac_dense!(fixed_handler::MakeParameter, cb::DenseCallback{T}, jac) where T
    jac[:,fixed_handler.fixed] .= zero(T)
end

function _eval_lag_hess_wrapper!(
    cb::DenseCallback,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractMatrix{T};
    obj_weight = one(T)
    ) where T

    hess_dense!(
        cb.nlp, x, y, hess;
        obj_weight=obj_weight * cb.obj_scale[]
    )

    _treat_fixed_variable_lag_hess_dense!(cb.fixed_handler, cb, hess)
end
function _treat_fixed_variable_lag_hess_dense!(fixed_handler::RelaxBound, cb::DenseCallback, hess) end
function _treat_fixed_variable_lag_hess_dense!(fixed_handler::MakeParameter, cb::DenseCallback{T}, hess) where T
    fixed = fixed_handler.fixed
    hess[:,fixed] .= zero(T)
    hess[fixed,:] .= zero(T)
    _set_diag!(hess, fixed, one(T))
end

function update_z!(cb, zl, zu, jacl)
    _update_z!(cb.fixed_handler, zl, zu, jacl, get_minimize(cb.nlp) ? 1 : -1)
end

function _update_z!(fixed_handler::MakeParameter, zl, zu, jacl, sense)
    zl_r = @view(zl[fixed_handler.fixed])
    zu_r = @view(zu[fixed_handler.fixed])
    jacl_r = @view(jacl[fixed_handler.fixed])
    map!(
        (x,y)->sense * max(x+y,0),
        zl_r,
        fixed_handler.grad_storage,
        jacl_r
    )
    map!(
        (x,y)->sense * max(-(x+y),0),
        zu_r,
        fixed_handler.grad_storage,
        jacl_r,
    )
end
function _update_z!(fixed_handler::RelaxBound, zl, zu, jacl, sense) end

function _set_diag!(A, inds, a)
    @inbounds @simd for i in inds
        A[i,i] = a
    end
end
