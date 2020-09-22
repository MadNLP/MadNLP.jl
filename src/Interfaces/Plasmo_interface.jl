# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

const dummy_function = ()->nothing

num_linkconstraints(modeledge::OptiEdge) = length(modeledge.linkconstraints)
moi_optimizer(modelnode::OptiNode) = modelnode.model.moi_backend.optimizer.model
function set_g_link!(linkedge::OptiEdge,l,gl,gu)
    cnt = 1
    for (ind,linkcon) in linkedge.linkconstraints
        l[cnt] = 0. # need to implement dual start later
        if linkcon.set isa MathOptInterface.EqualTo
            gl[cnt] = linkcon.set.value
            gu[cnt] = linkcon.set.value
        elseif linkcon.set isa MathOptInterface.GreaterThan
            gl[cnt] = linkcon.set.lower
            gu[cnt] = Inf
        elseif linkcon.set isa MathOptInterface.LessThan
            gl[cnt] = -Inf
            gu[cnt] = linkcon.set.upper
        else
            gl[cnt] = linkcon.set.lower
            gu[cnt] = linkcon.set.upper
        end
        cnt += 1
    end
end

function hessian_lagrangian_structure(graph::OptiGraph,I,J,ninds,nnzs_hess_inds,modelnodes)
    @blas_safe_threads for k=1:length(modelnodes)
        isempty(nnzs_hess_inds[k]) && continue
        offset = ninds[k][1]-1
        II = view(I,nnzs_hess_inds[k])
        JJ = view(J,nnzs_hess_inds[k])
        hessian_lagrangian_structure(
            moi_optimizer(modelnodes[k]),II,JJ)
        II.+= offset
        JJ.+= offset
    end
end

function jacobian_structure(linkedge::OptiEdge,I,J,ninds,x_index_map,g_index_map)
    offset=1
    for linkcon in getlinkconstraints(linkedge)
        offset += jacobian_structure(linkcon,I,J,ninds,x_index_map,g_index_map,offset)
    end
end
function jacobian_structure(linkcon,I,J,ninds,x_index_map,g_index_map,offset)
    cnt = 0
    for var in get_vars(linkcon)
        I[offset+cnt] = g_index_map[linkcon]
        J[offset+cnt] = x_index_map[var]
        cnt += 1
    end
    return cnt
end

function jacobian_structure(
    graph::OptiGraph,I,J,ninds,minds,pinds,nnzs_jac_inds,nnzs_link_jac_inds,
    x_index_map,g_index_map,modelnodes,linkedges)
    
    @blas_safe_threads for k=1:length(modelnodes)
        isempty(nnzs_jac_inds[k]) && continue
        offset_i = minds[k][1]-1
        offset_j = ninds[k][1]-1
        II = view(I,nnzs_jac_inds[k])
        JJ = view(J,nnzs_jac_inds[k])
        jacobian_structure(
            moi_optimizer(modelnodes[k]),II,JJ)
        II.+= offset_i 
        JJ.+= offset_j 
    end
    
    @blas_safe_threads for q=1:length(linkedges)
        isempty(nnzs_link_jac_inds[q]) && continue
        II = view(I,nnzs_link_jac_inds[q])
        JJ = view(J,nnzs_link_jac_inds[q])
        jacobian_structure(
            linkedges[q],II,JJ,ninds,x_index_map,g_index_map)
    end
end
function eval_objective(graph::OptiGraph,x,ninds,x_index_map,modelnodes)
    obj = Threads.Atomic{Float64}(0.)
    @blas_safe_threads for k=1:length(modelnodes)
         Threads.atomic_add!(obj,eval_objective(
             moi_optimizer(modelnodes[k]),view(x,ninds[k])))
    end
    return obj.value + eval_function(graph.objective_function,x,ninds,x_index_map)
end
function eval_objective_gradient(graph::OptiGraph,f,x,ninds,modelnodes)
    @blas_safe_threads for k=1:length(modelnodes)
        eval_objective_gradient(moi_optimizer(modelnodes[k]),view(f,ninds[k]),view(x,ninds[k]))
    end
end

function eval_function(aff::GenericAffExpr,x,ninds,x_index_map)
    function_value = aff.constant
    for (var,coef) in aff.terms
        function_value += coef*x[x_index_map[var]]
    end
    return function_value
end
function eval_constraint(linkedge::OptiEdge,c,x,ninds,x_index_map)
    cnt = 1
    for linkcon in getlinkconstraints(linkedge)
        c[cnt] = eval_function(get_func(linkcon),x,ninds,x_index_map)
        cnt += 1
    end
end
get_func(linkcon) = linkcon.func
function eval_constraint(graph::OptiGraph,c,x,ninds,minds,pinds,x_index_map,modelnodes,linkedges)
    @blas_safe_threads for k=1:length(modelnodes)
        eval_constraint(moi_optimizer(modelnodes[k]),view(c,minds[k]),view(x,ninds[k]))
    end
    @blas_safe_threads for q=1:length(linkedges)
        eval_constraint(linkedges[q],view(c,pinds[q]),x,ninds,x_index_map)
    end
end

function eval_hessian_lagrangian(graph::OptiGraph,hess,x,sig,l,
                                 ninds,minds,nnzs_hess_inds,modelnodes)
    @blas_safe_threads for k=1:length(modelnodes)
        isempty(nnzs_hess_inds) && continue
        eval_hessian_lagrangian(moi_optimizer(modelnodes[k]),
                                view(hess,nnzs_hess_inds[k]),view(x,ninds[k]),sig,
                                view(l,minds[k]))
    end
end

function eval_constraint_jacobian(linkedge::OptiEdge,jac,x)
    offset=0
    for linkcon in getlinkconstraints(linkedge)
        offset+=eval_constraint_jacobian(linkcon,jac,offset)
    end
end
function eval_constraint_jacobian(linkcon,jac,offset)
    cnt = 0
    for coef in get_coeffs(linkcon)
        cnt += 1
        jac[offset+cnt] = coef
    end
    return cnt
end
get_vars(linkcon) = keys(linkcon.func.terms)
get_coeffs(linkcon) = values(linkcon.func.terms)
          
function eval_constraint_jacobian(graph::OptiGraph,jac,x,
                                  ninds,minds,nnzs_jac_inds,nnzs_link_jac_inds,modelnodes,linkedges)
    @blas_safe_threads for k=1:length(modelnodes)
        eval_constraint_jacobian(
            moi_optimizer(modelnodes[k]),view(jac,nnzs_jac_inds[k]),view(x,ninds[k]))
    end
    @blas_safe_threads for q=1:length(linkedges)
        eval_constraint_jacobian(linkedges[q],view(jac,nnzs_link_jac_inds[q]),x)
    end
end
get_nnz_link_jac(linkedge::OptiEdge) = sum(
    length(linkcon.func.terms) for (ind,linkcon) in linkedge.linkconstraints)


function NonlinearProgram(graph::OptiGraph)
    modelnodes = all_nodes(graph)
    linkedges = all_edges(graph)

    for modelnode in modelnodes
        num_variables(modelnode) == 0 && error("Empty node exist! Delete the empty nodes.")
    end
    
    @blas_safe_threads for k=1:length(modelnodes)
        set_optimizer(modelnodes[k].model,Optimizer)
        if modelnodes[k].model.nlp_data !== nothing
            MOI.set(modelnodes[k].model, MOI.NLPBlock(),
                    _create_nlp_block_data(modelnodes[k].model))
            empty!(modelnodes[k].model.nlp_data.nlconstr_duals)
        end
        MOIU.attach_optimizer(modelnodes[k].model)
        MOI.initialize(moi_optimizer(modelnodes[k]).nlp_data.evaluator,[:Grad,:Hess,:Jac])
    end

    K = length(modelnodes)
    ns= [num_variables(moi_optimizer(modelnode)) for modelnode in modelnodes]
    n = sum(ns)
    ns_cumsum = cumsum(ns)
    ms= [num_constraints(moi_optimizer(modelnode)) for modelnode in modelnodes]
    ms_cumsum = cumsum(ms)
    m = sum(ms)

    nnzs_hess = [get_nnz_hess(moi_optimizer(modelnode)) for modelnode in modelnodes]
    nnzs_hess_cumsum = cumsum(nnzs_hess)
    nnz_hess = sum(nnzs_hess)

    nnzs_jac = [get_nnz_jac(moi_optimizer(modelnode)) for modelnode in modelnodes]
    nnzs_jac_cumsum = cumsum(nnzs_jac)
    nnz_jac = sum(nnzs_jac)

    nnzs_link_jac = [get_nnz_link_jac(linkedge) for linkedge in linkedges]
    nnzs_link_jac_cumsum = cumsum(nnzs_link_jac)
    nnz_link_jac = isempty(nnzs_link_jac) ? 0 : sum(nnzs_link_jac)

    ninds = [(i==1 ? 0 : ns_cumsum[i-1])+1:ns_cumsum[i] for i=1:K]
    minds = [(i==1 ? 0 : ms_cumsum[i-1])+1:ms_cumsum[i] for i=1:K]
    nnzs_hess_inds = [(i==1 ? 0 : nnzs_hess_cumsum[i-1])+1:nnzs_hess_cumsum[i] for i=1:K]
    nnzs_jac_inds = [(i==1 ? 0 : nnzs_jac_cumsum[i-1])+1:nnzs_jac_cumsum[i] for i=1:K]
    
    Q = length(linkedges)
    ps= [num_linkconstraints(modeledge) for modeledge in linkedges]
    ps_cumsum =  cumsum(ps)
    p = sum(ps)
    pinds = [(i==1 ? m : m+ps_cumsum[i-1])+1:m+ps_cumsum[i] for i=1:Q]
    nnzs_link_jac_inds =
        [(i==1 ? nnz_jac : nnz_jac+nnzs_link_jac_cumsum[i-1])+1: nnz_jac + nnzs_link_jac_cumsum[i] for i=1:Q]
    
    x =Vector{Float64}(undef,n)
    g =Vector{Float64}(undef,m+p)

    xl=Vector{Float64}(undef,n)
    xu=Vector{Float64}(undef,n)
    zl=Vector{Float64}(undef,n)
    zu=Vector{Float64}(undef,n)

    l =Vector{Float64}(undef,m+p)
    gl=Vector{Float64}(undef,m+p)
    gu=Vector{Float64}(undef,m+p)

    @blas_safe_threads for k=1:K
        set_x!(moi_optimizer(modelnodes[k]),view(x,ninds[k]),view(xl,ninds[k]),
               view(xu,ninds[k]),view(zl,ninds[k]),view(zu,ninds[k]))
        set_g!(moi_optimizer(modelnodes[k]),view(l,minds[k]),view(gl,minds[k]),view(gu,minds[k]))
    end

    @blas_safe_threads for q=1:Q
        set_g_link!(linkedges[q],view(l,pinds[q]),view(gl,pinds[q]),view(gu,pinds[q]))
    end

    modelmap=Dict(modelnodes[k].model=> k for k=1:K)
    x_index_map = Dict(
        var=>ninds[modelmap[var.model]][backend(var.model).model_to_optimizer_map[var.index].value]
        for modelnode in modelnodes for var in all_variables(modelnode))
    cnt = 0
    g_index_map = Dict(con=> m + (cnt+=1) for linkedge in linkedges for (ind,con) in linkedge.linkconstraints)

    obj(x) =  eval_objective(graph,x,ninds,x_index_map,modelnodes)
    obj_grad!(f,x) =eval_objective_gradient(graph,f,x,ninds,modelnodes)
    con!(c,x) = eval_constraint(graph,c,x,ninds,minds,pinds,x_index_map,modelnodes,linkedges)
    lag_hess!(hess,x,l,sig::Float64) =eval_hessian_lagrangian(
        graph,hess,x,sig,l,ninds,minds,nnzs_hess_inds,modelnodes)
    con_jac!(jac,x)=eval_constraint_jacobian(
        graph,jac,x,ninds,minds,nnzs_jac_inds,nnzs_link_jac_inds,modelnodes,linkedges)
    hess_sparsity!(I,J)=hessian_lagrangian_structure(graph,I,J,ninds,nnzs_hess_inds,modelnodes)
    jac_sparsity!(I,J) =jacobian_structure(
        graph,I,J,ninds,minds,pinds,nnzs_jac_inds,nnzs_link_jac_inds,x_index_map,g_index_map,modelnodes,linkedges)

    # some of outputs should be set automatically with this
    @blas_safe_threads for k=1:K
        moi_optimizer(modelnodes[k]).nlp = NonlinearProgram(
            0,0,0,0,0.,view(x,ninds[k]),Float64[],view(l,minds[k]),view(zl,ninds[k]),view(zu,ninds[k]),
            Float64[],Float64[],Float64[],Float64[],
            dummy_function,dummy_function,dummy_function,dummy_function,dummy_function,
            dummy_function,dummy_function,:Initial, Dict{Symbol,Any}())
    end

    jac_constant = true
    hess_constant = true
    for node in modelnodes
        j,h = is_jac_hess_constant(moi_optimizer(node))
        jac_constant = jac_constant & j
        hess_constant = hess_constant & h
    end

    ext = Dict{Symbol,Any}(:n=>n,:m=>m,:p=>p,:ninds=>ninds,:minds=>minds,:pinds=>pinds,
                           :linkedges=>linkedges,:jac_constant=>jac_constant,:hess_constant=>hess_constant)
    
    return NonlinearProgram(n,m+p,nnz_hess,nnz_jac+nnz_link_jac,0.,x,g,l,zl,zu,xl,xu,gl,gu,obj,obj_grad!,
                            con!,con_jac!,lag_hess!,hess_sparsity!,jac_sparsity!, :Initial , ext)
end

function get_part(graph,nlp::NonlinearProgram)
    n = nlp.ext[:n]
    m = nlp.ext[:m]
    p = nlp.ext[:p]
    
    ninds = nlp.ext[:ninds]
    minds = nlp.ext[:minds]
    pinds = nlp.ext[:pinds]
    
    ind_ineq = findall(nlp.gl.!=nlp.gu)
    l = length(ind_ineq)

    part = Vector{Int}(undef,n+m+l+p)
    
    for k=1:length(ninds)
        part[ninds[k]].=k
    end
    for k=1:length(minds)
        part[minds[k].+n.+l].=k
    end
    cnt = 0

    for linkedge in nlp.ext[:linkedges]
        for (ind,con) in linkedge.linkconstraints
            cnt+=1
            attached_node_idx = graph.node_idx_map[con.attached_node]
            part[n+l+m+cnt] = attached_node_idx != nothing ? attached_node_idx : error("All the link constraints need to be attached to a node")
        end
    end

    cnt = 0
    for q in ind_ineq
        cnt+=1
        part[n+cnt] = part[n+l+q]
    end
    
    
    return part
end

function optimize!(graph::OptiGraph; option_dict = Dict{Symbol,Any}(), kwargs...)
    nlp = NonlinearProgram(graph)
    
    
    if (haskey(kwargs,:schwarz_custom_partition) && kwargs[:schwarz_custom_partition]) ||
        (haskey(option_dict,:schwarz_custom_partition) && option_dict[:schwarz_custom_partition])
        
        part= get_part(graph,nlp)
        option_dict[:schwarz_part] = part
        option_dict[:schwarz_num_parts] = num_all_nodes(graph)
        
    elseif (haskey(kwargs,:schur_custom_partition) && kwargs[:schur_custom_partition]) ||
        (haskey(option_dict,:schur_custom_partition) && option_dict[:schur_custom_partition])

        part= get_part(graph,nlp)
        K = num_all_nodes(graph) 
        part[part.>K].=0
        option_dict[:schur_part] = part
        option_dict[:schur_num_parts] = K
    end
    
    option_dict[:jacobian_constant] = nlp.ext[:jac_constant]
    option_dict[:hessian_constant] = nlp.ext[:hess_constant]
    graph.optimizer = Solver(nlp;option_dict=option_dict,kwargs...)
        
    optimize!(graph.optimizer)
    @blas_safe_threads for modelnode in all_nodes(graph) # set outputs
        moi_optimizer(modelnode).nlp.status = graph.optimizer.status
        moi_optimizer(modelnode).nlp.obj_val= graph.optimizer.obj_val
    end
end

termination_status(ips::Solver) = termination_status(ips.nlp.status)
