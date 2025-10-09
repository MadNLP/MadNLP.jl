# API for getting internal members of IPM
# The idea here is that a subtype of AbstractMadNLPSolver may specialize
# some or all of these and the core functionality of the IPM in MadNLP
# should function as normal.

# getters for all fields
for (k, attribute) in enumerate(fieldnames(MadNLPSolver))
    fname = "_$(attribute)"
    @eval begin
        @inline function $(Symbol(fname))(solver::MadNLPSolver)
            return getfield(solver, $k)
        end
    end
end

# Necessary Mutators
_RR!(solver::AbstractMadNLPSolver, rhs) = solver.RR = rhs
_n!(solver::AbstractMadNLPSolver, rhs::Int) = solver.n = rhs
_m!(solver::AbstractMadNLPSolver, rhs::Int) = solver.m = rhs
_nlb!(solver::AbstractMadNLPSolver, rhs::Int) = solver.nlb = rhs
_nub!(solver::AbstractMadNLPSolver, rhs::Int) = solver.nub = rhs
_status!(solver::AbstractMadNLPSolver, rhs::Status) = solver.status = rhs
_mu!(solver::AbstractMadNLPSolver{T}, rhs::T) where{T} = solver.mu = rhs
_obj_val!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.obj_val = rhs
_obj_val_trial!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.obj_val_trial = rhs
_inf_pr!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.inf_pr = rhs
_inf_du!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.inf_du = rhs
_inf_compl!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.inf_compl = rhs
_theta_min!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.theta_min = rhs
_theta_max!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.theta_max = rhs
_tau!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.tau = rhs
_alpha!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.alpha = rhs
_alpha_z!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.alpha_z = rhs
_ftype!(solver::AbstractMadNLPSolver, rhs::String) = solver.ftype = rhs
_del_w!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.del_w = rhs
_del_w_last!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.del_w_last = rhs
_del_c!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.del_c = rhs

# Computed quantities
# TODO(@anton) I now think these would be useful to have, but which quantities are wrapped this way is a good question
_theta(solver::AbstractMadNLPSolver) = get_theta(_c(solver))
_varphi(solver::AbstractMadNLPSolver) = get_varphi(_obj_val(solver), _x_lr(solver), _xl_r(solver), _xu_r(solver), _x_ur(solver), _mu(solver))
_kkt_error(solver::AbstractMadNLPSolver) = max(_inf_pr(solver), _inf_du(solver), _inf_compl(solver))
_inf_compl(solver::AbstractMadNLPSolver{T}, sc::T; mu=_mu(solver)) where {T} = get_inf_compl(_x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver), mu, sc)
_inf_total(solver::AbstractMadNLPSolver) = max(_inf_pr(solver),_inf_du(solver),_inf_compl(solver))
