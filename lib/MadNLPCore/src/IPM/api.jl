# API for getting internal members of IPM
# The idea here is that a subtype of AbstractMadNLPSolver may specialize
# some or all of these and the core functionality of the IPM in MadNLP
# should function as normal.

# getters for all fields
for (k, attribute) in enumerate(fieldnames(MadNLPSolver))
    fname = "get_$(attribute)"
    @eval begin
        @inline function $(Symbol(fname))(solver::MadNLPSolver)
            return getfield(solver, $k)
        end
    end
end

# Necessary Mutators
for (k, (attribute, type)) in enumerate(zip(fieldnames(MadNLPSolver), fieldtypes(MadNLPSolver)))
    # These types shouldn't ever be set by value so we ignore them.
    if type <: Union{AbstractVector,
                     PrimalVector,
                     AbstractKKTVector,
                     MadNLPOptions,
                     MadNLPCounters,
                     MadNLPLogger,
                     AbstractKKTSystem,
                     AbstractNLPModel,
                     AbstractCallback,
                     AbstractIterator,
                     AbstractInertiaCorrector}
      continue
    end
    fname = "set_$(attribute)!"
    @eval begin
        @inline function $(Symbol(fname))(solver::MadNLPSolver, rhs)
            return setfield!(solver, $k, rhs)
        end
    end
end

# Computed quantities
# TODO(@anton) I now think these would be useful to have, but which quantities are wrapped this way is a good question
get_theta(solver::AbstractMadNLPSolver) = get_theta(get_c(solver))
get_varphi(solver::AbstractMadNLPSolver) = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), get_mu(solver))
get_kkt_error(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver), get_inf_du(solver), get_inf_compl(solver))
get_inf_compl(solver::AbstractMadNLPSolver{T}, sc::T; mu=get_mu(solver)) where {T} = get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver), mu, sc)
get_inf_barrier(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver),get_inf_du(solver),get_inf_compl_mu(solver))
get_inf_total(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver),get_inf_du(solver),get_inf_compl(solver))
