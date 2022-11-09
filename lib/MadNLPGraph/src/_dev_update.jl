using Plasmo
using MadNLP
using MadNLPGraph

graph = OptiGraph()
@optinode(graph,nodes[1:4])

nl_refs = []
for node in nodes
    @variable(node,x>=0)
    @variable(node,y>=0)
    @constraint(node,x + y >= 3)
    ref = @NLconstraint(node,x^3 >= 1)
    push!(nl_refs,ref)
end
@objective(graph,Min,sum(n[:x] for n in nodes))
@linkconstraint(graph,sum(node[:y] for node in nodes) == 10)

MadNLPGraph.optimize!(graph)
# GraphModel(graph)

# # TODO: be more robust with how we get the underlying MadNLP optimizer
# function _caching_optimizer(optinode::OptiNode)
#     if isa(optinode.model.moi_backend, MOIU.CachingOptimizer)
#         return optinode.model.moi_backend
#     else
#         return optinode.model.moi_backend.optimizer
#     end
# end
# _caching_optimizer(model::Any) = model.moi_backend
#
# # inner_optimizer = MadNLP.Optimizer
# inner_optimizer(optinode::OptiNode) = _caching_optimizer(optinode).optimizer.model
#
# get_nnz_link_jac(linkedge::OptiEdge) = sum(
#     length(linkcon.func.terms) for (ind,linkcon) in linkedge.linkconstraints
#     )
#
# # set initial dual, lower, and upper linkconstraint bounds
# function set_g_link!(linkedge::OptiEdge, l, gl, gu)
#     cnt = 1
#     for (ind,linkcon) in linkedge.linkconstraints
#         l[cnt] = 0. # need to implement dual start later
#         if linkcon.set isa MOI.EqualTo
#             gl[cnt] = linkcon.set.value
#             gu[cnt] = linkcon.set.value
#         elseif linkcon.set isa MOI.GreaterThan
#             gl[cnt] = linkcon.set.lower
#             gu[cnt] = Inf
#         elseif linkcon.set isa MOI.LessThan
#             gl[cnt] = -Inf
#             gu[cnt] = linkcon.set.upper
#         else
#             gl[cnt] = linkcon.set.lower
#             gu[cnt] = linkcon.set.upper
#         end
#         cnt += 1
#     end
# end
#
#
# # function GraphModel(graph::OptiGraph)
# optinodes = all_nodes(graph)
# linkedges = all_edges(graph)
#
# for optinode in optinodes
#     num_variables(optinode) == 0 && error("MadNLP does not support empty nodes. You will need to create your optigraph without empty optinodes.")
# end
#
# moi_models = Vector{MadNLP.MOIModel}(undef,length(optinodes))
# #create a MadNLP optimizer on each optinode
# for k=1:length(optinodes)
#     optinode = optinodes[k]
#     # set constructor; optimizer not yet attached
#     set_optimizer(optinode, MadNLP.Optimizer)
#     # Initialize NLP evaluators on each node
#     # In JuMP, this gets set before calling `MOI.optimize!`
#     jump_model = Plasmo.jump_model(optinode)
#     nonlinear_model = JuMP.nonlinear_model(jump_model)
#     if nonlinear_model !== nothing
#         evaluator = MOI.Nonlinear.Evaluator(
#             nonlinear_model,
#             MOI.Nonlinear.SparseReverseMode(),
#             JuMP.index.(JuMP.all_variables(jump_model)),
#         )
#         MOI.set(jump_model, MOI.NLPBlock(), MOI.NLPBlockData(evaluator))
#         # TODO: check whether we still need to do this
#         # empty!(modelnodes[k].model.nlp_data.nlconstr_duals)
#         empty!(optinode.nlp_duals)
#     end
#     # attach optimizer on node. this populates the madnlp optimizer
#     MOIU.attach_optimizer(jump_model)
#
#     madnlp_optimizer = inner_optimizer(optinode)
#     moi_models[k] = MadNLP.MOIModel(madnlp_optimizer)
#
#     # initialize optinode evaluator
#     # madnlp_model = inner_optimizer(optinode)
#     # MOI.initialize(madnlp_model.nlp_data.evaluator, [:Grad,:Hess,:Jac])
# end
#
# # calculate dimensions
# K = length(optinodes)
# ns= [num_variables(optinode) for optinode in optinodes]
# n = sum(ns)
# ns_cumsum = cumsum(ns)
# ms= [num_constraints(optinode) for optinode in optinodes]
# ms_cumsum = cumsum(ms)
# m = sum(ms)
#
# nlps = [moi.model for moi in moi_models]
# # hessian nonzeros
# nnzs_hess = [model.meta.nnzh for model in moi_models]
# nnzs_hess_cumsum = cumsum(nnzs_hess)
# nnz_hess = sum(nnzs_hess)
#
# #nnzs_jac = [length(MOI.jacobian_structure(nlp)) for nlp in nlps]
# nnzs_jac = [model.meta.nnzj for model in moi_models]
# nnzs_jac_cumsum = cumsum(nnzs_jac)
# nnz_jac = sum(nnzs_jac)
#
# # link jacobian nonzeros
# nnzs_link_jac = [get_nnz_link_jac(linkedge) for linkedge in linkedges]
# nnzs_link_jac_cumsum = cumsum(nnzs_link_jac)
# nnz_link_jac = isempty(nnzs_link_jac) ? 0 : sum(nnzs_link_jac)
#
# ninds = [(i==1 ? 0 : ns_cumsum[i-1])+1:ns_cumsum[i] for i=1:K]
# minds = [(i==1 ? 0 : ms_cumsum[i-1])+1:ms_cumsum[i] for i=1:K]
# nnzs_hess_inds = [(i==1 ? 0 : nnzs_hess_cumsum[i-1])+1:nnzs_hess_cumsum[i] for i=1:K]
# nnzs_jac_inds = [(i==1 ? 0 : nnzs_jac_cumsum[i-1])+1:nnzs_jac_cumsum[i] for i=1:K]
#
# Q = length(linkedges)
# ps= [Plasmo.num_linkconstraints(modeledge) for modeledge in linkedges]
# ps_cumsum =  cumsum(ps)
# p = sum(ps)
# pinds = [(i==1 ? m : m+ps_cumsum[i-1])+1:m+ps_cumsum[i] for i=1:Q]
# nnzs_link_jac_inds =
#     [(i==1 ? nnz_jac : nnz_jac+nnzs_link_jac_cumsum[i-1])+1: nnz_jac + nnzs_link_jac_cumsum[i] for i=1:Q]
#
# # primals, lower, upper bounds
# x = Vector{Float64}(undef,n)
# xl = Vector{Float64}(undef,n)
# xu = Vector{Float64}(undef,n)
#
# # duals, constraints lower, upper bounds
# l = Vector{Float64}(undef,m+p)
# gl = Vector{Float64}(undef,m+p)
# gu = Vector{Float64}(undef,m+p)
#
# # set start values for variables and constraints
# for k=1:K
#     model = moi_models[k]
#     x[ninds[k]] .= model.meta.x0
#     xl[ninds[k]] .= model.meta.lvar
#     xu[ninds[k]] .= model.meta.uvar
#
#     l[minds[k]] .= model.meta.y0
#     gl[minds[k]] .= model.meta.lcon
#     gu[minds[k]] .= model.meta.ucon
# end
#
# # set link constraint values
# for q=1:Q
#     set_g_link!(linkedges[q], view(l,pinds[q]), view(gl,pinds[q]), view(gu,pinds[q]))
# end
#
# # map variables to graph indices. used for link jacobians
# x_index_map = Dict()
# for k = 1:K
#     optinode = optinodes[k]
#     for var in all_variables(optinode)
#         x_index_map[var] = ninds[k][JuMP.index(var).value]
#     end
# end
#
# g_index_map = Dict()
# cnt = 1
# for linkref in all_linkconstraints(graph)
#     con = constraint_object(linkref)
#     g_index_map[con] = m + cnt
# end
#
# jac_constant = true
# hess_constant = true
# for moi in moi_models
#     jac_constant = jac_constant & moi.model.options[:jacobian_constant]
#     hess_constant = hess_constant & moi.model.options[:hessian_constant]
# end
#
# ext = Dict{Symbol,Any}(
#     :n=>n, :m=>m,:p=>p,
#     :ninds=>ninds, :minds=>minds, :pinds=>pinds,
#     :linkedges=>linkedges, :jac_constant=>jac_constant, :hess_constant=>hess_constant
# )
