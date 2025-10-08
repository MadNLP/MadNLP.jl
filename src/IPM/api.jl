# API for getting internal members of IPM
# The idea here is that a subtype of AbstractMadNLPSolver may specialize
# some or all of these and the core functionality of the IPM in MadNLP
# should function as normal.
# TODO(@anton) We could in principle write a macro to define this "automagically"

# Solver data
_kkt(solver::AbstractMadNLPSolver) = solver.kkt
_nlp(solver::AbstractMadNLPSolver) = solver.nlp
_cb(solver::AbstractMadNLPSolver) = solver.cb
_cnt(solver::AbstractMadNLPSolver) = solver.cnt
_logger(solver::AbstractMadNLPSolver) = solver.logger
_iterator(solver::AbstractMadNLPSolver) = solver.iterator
_RR(solver::AbstractMadNLPSolver) = solver.RR
_RR!(solver::AbstractMadNLPSolver, rhs) = solver.RR = rhs
_output(solver::AbstractMadNLPSolver) = solver.output
_inertia_corrector(solver::AbstractMadNLPSolver) = solver.inertia_corrector

# Solver sizes
_n(solver::AbstractMadNLPSolver) = solver.n
_m(solver::AbstractMadNLPSolver) = solver.m
_nlb(solver::AbstractMadNLPSolver) = solver.nlb
_nub(solver::AbstractMadNLPSolver) = solver.nub

# Solver size mutators
_n!(solver::AbstractMadNLPSolver, rhs::Int) = solver.n = rhs
_m!(solver::AbstractMadNLPSolver, rhs::Int) = solver.m = rhs
_nlb!(solver::AbstractMadNLPSolver, rhs::Int) = solver.nlb = rhs
_nub!(solver::AbstractMadNLPSolver, rhs::Int) = solver.nub = rhs

# Vectors
_x(solver::AbstractMadNLPSolver) = solver.x
_y(solver::AbstractMadNLPSolver) = solver.y
_zl(solver::AbstractMadNLPSolver) = solver.zl
_xl(solver::AbstractMadNLPSolver) = solver.xl
_zu(solver::AbstractMadNLPSolver) = solver.zu
_xu(solver::AbstractMadNLPSolver) = solver.xu
_f(solver::AbstractMadNLPSolver) = solver.f
_c(solver::AbstractMadNLPSolver) = solver.c
_jacl(solver::AbstractMadNLPSolver) = solver.jacl
_d(solver::AbstractMadNLPSolver) = solver.d
_p(solver::AbstractMadNLPSolver) = solver.p
__w1(solver::AbstractMadNLPSolver) = solver._w1
__w2(solver::AbstractMadNLPSolver) = solver._w2
__w3(solver::AbstractMadNLPSolver) = solver._w3
__w4(solver::AbstractMadNLPSolver) = solver._w4
_x_trial(solver::AbstractMadNLPSolver) = solver.x_trial
_c_trial(solver::AbstractMadNLPSolver) = solver.c_trial
_c_slk(solver::AbstractMadNLPSolver) = solver.c_slk
_rhs(solver::AbstractMadNLPSolver) = solver.rhs
_ind_ineq(solver::AbstractMadNLPSolver) = solver.ind_ineq
_ind_fixed(solver::AbstractMadNLPSolver) = solver.ind_fixed
_ind_llb(solver::AbstractMadNLPSolver) = solver.ind_llb
_ind_uub(solver::AbstractMadNLPSolver) = solver.ind_uub
_x_lr(solver::AbstractMadNLPSolver) = solver.x_lr
_x_ur(solver::AbstractMadNLPSolver) = solver.x_ur
_xl_r(solver::AbstractMadNLPSolver) = solver.xl_r
_xu_r(solver::AbstractMadNLPSolver) = solver.xu_r
_zl_r(solver::AbstractMadNLPSolver) = solver.zl_r
_zu_r(solver::AbstractMadNLPSolver) = solver.zu_r
_dx_lr(solver::AbstractMadNLPSolver) = solver.dx_lr
_dx_ur(solver::AbstractMadNLPSolver) = solver.dx_ur
_x_trial_lr(solver::AbstractMadNLPSolver) = solver.x_trial_lr
_x_trial_ur(solver::AbstractMadNLPSolver) = solver.x_trial_ur

# Options
_opt(solver::AbstractMadNLPSolver) = solver.opt

# Solver state
_status(solver::AbstractMadNLPSolver) = solver.status
_mu(solver::AbstractMadNLPSolver) = solver.mu
_filter(solver::AbstractMadNLPSolver) = solver.filter
_obj_val(solver::AbstractMadNLPSolver) = solver.obj_val
_obj_val_trial(solver::AbstractMadNLPSolver) = solver.obj_val_trial
_inf_pr(solver::AbstractMadNLPSolver) = solver.inf_pr
_inf_du(solver::AbstractMadNLPSolver) = solver.inf_du
_inf_compl(solver::AbstractMadNLPSolver) = solver.inf_compl
_theta_min(solver::AbstractMadNLPSolver) = solver.theta_min
_theta_max(solver::AbstractMadNLPSolver) = solver.theta_max
_tau(solver::AbstractMadNLPSolver) = solver.tau
_alpha(solver::AbstractMadNLPSolver) = solver.alpha
_alpha_z(solver::AbstractMadNLPSolver) = solver.alpha_z
_ftype(solver::AbstractMadNLPSolver) = solver.ftype
_del_w(solver::AbstractMadNLPSolver) = solver.del_w
_del_w_last(solver::AbstractMadNLPSolver) = solver.del_w_last
_del_c(solver::AbstractMadNLPSolver) = solver.del_c

# Solver state mutators
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
