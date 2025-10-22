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
set_inf_compl_mu!(solver::AbstractMadNLPSolver{T}, rhs::T) where {T} = solver.inf_compl_mu = rhs
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
get_theta(solver::AbstractMadNLPSolver) = get_theta(get_c(solver))
get_varphi(solver::AbstractMadNLPSolver) = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), get_mu(solver))
get_kkt_error(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver), get_inf_du(solver), get_inf_compl(solver))
get_inf_compl(solver::AbstractMadNLPSolver{T}, sc::T; mu=get_mu(solver)) where {T} = get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver), mu, sc)
get_inf_barrier(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver),get_inf_du(solver),get_inf_compl_mu(solver))
get_inf_total(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver),get_inf_du(solver),get_inf_compl(solver))
