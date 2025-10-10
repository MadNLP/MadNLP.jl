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
set_RR!(solver::AbstractMadNLPSolver, rhs) = solver.RR = rhs
set_n!(solver::AbstractMadNLPSolver, rhs::Int) = solver.n = rhs
set_m!(solver::AbstractMadNLPSolver, rhs::Int) = solver.m = rhs
set_nlb!(solver::AbstractMadNLPSolver, rhs::Int) = solver.nlb = rhs
set_nub!(solver::AbstractMadNLPSolver, rhs::Int) = solver.nub = rhs
set_status!(solver::AbstractMadNLPSolver, rhs::Status) = solver.status = rhs
set_mu!(solver::AbstractMadNLPSolver{T}, rhs::T) where{T} = solver.mu = rhs
set_obj_val!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.obj_val = rhs
set_obj_val_trial!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.obj_val_trial = rhs
set_inf_pr!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.inf_pr = rhs
set_inf_du!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.inf_du = rhs
set_inf_compl!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.inf_compl = rhs
set_theta_min!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.theta_min = rhs
set_theta_max!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.theta_max = rhs
set_tau!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.tau = rhs
set_alpha!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.alpha = rhs
set_alpha_z!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.alpha_z = rhs
set_ftype!(solver::AbstractMadNLPSolver, rhs::String) = solver.ftype = rhs
set_del_w!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.del_w = rhs
set_del_w_last!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.del_w_last = rhs
set_del_c!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.del_c = rhs

# Computed quantities
# TODO(@anton) I now think these would be useful to have, but which quantities are wrapped this way is a good question
_theta(solver::AbstractMadNLPSolver) = get_theta(_c(solver))
_varphi(solver::AbstractMadNLPSolver) = get_varphi(_obj_val(solver), _x_lr(solver), _xl_r(solver), _xu_r(solver), _x_ur(solver), _mu(solver))
_kkt_error(solver::AbstractMadNLPSolver) = max(_inf_pr(solver), _inf_du(solver), _inf_compl(solver))
_inf_compl(solver::AbstractMadNLPSolver{T}, sc::T; mu=_mu(solver)) where {T} = get_inf_compl(_x_lr(solver),_xl_r(solver),_zl_r(solver),_xu_r(solver),_x_ur(solver),_zu_r(solver), mu, sc)
_inf_total(solver::AbstractMadNLPSolver) = max(_inf_pr(solver),_inf_du(solver),_inf_compl(solver))
