atol = rtol = 1e-8
n_nodes=2
M = 100
d = sin.((0:M*n_nodes).*pi/100)
    
graph = OptiGraph()
@optinode(graph,nodes[1:n_nodes])

#Node models
nodecnt = 1
for (i,node) in enumerate(nodes)
    local x
    global nodecnt
    @variable(node, x[1:M])
    @variable(node, -1<=u[1:M]<=1)
    @constraint(node, dynamics[i in 1:M-1], x[i+1] == x[i] + u[i])
    @objective(node, Min, sum(x[i]^2 - 2*x[i]*d[i+(nodecnt-1)*M] for i in 1:M) +
               sum(u[i]^2 for i in 1:M))
    nodecnt += 1
end
n1 = getnode(graph,1)
@constraint(n1,n1[:x][1] == 0)
for i=1:n_nodes-1
    @linkconstraint(graph, nodes[i+1][:x][1] == nodes[i][:x][M] + nodes[i][:u][M],attach=nodes[i+1])
end

#First node gets initial condition
kwargs_collection = [
    Dict(:print_level=>MadNLP.ERROR),
    Dict(:linear_solver=>"schur",  :schur_custom_partition=>true,:print_level=>MadNLP.ERROR),
    Dict(:linear_solver=>"schwarz",:schwarz_custom_partition=>true,:print_level=>MadNLP.ERROR)
]
@testset "Plasmo test" for kwargs in kwargs_collection
    MadNLP.optimize!(graph;kwargs...);
    @test termination_status(graph.optimizer) == MOI.LOCALLY_SOLVED 
end

optimizer_constructors = [
    ()->MadNLP.Optimizer(print_level=MadNLP.ERROR),
    ()->MadNLP.Optimizer(linear_solver="schur",schur_num_parts=2,print_level=MadNLP.ERROR),
    ()->MadNLP.Optimizer(linear_solver="schwarz",schwarz_num_parts=2),
    ()->MadNLP.Optimizer(linear_solver="schwarz",schwarz_num_parts_upper=2,schwarz_num_parts=10)
]

@testset "Plasmo test" for optimizer_constructor in optimizer_constructors
    node,~=combine(graph)
    m = node.model
    set_optimizer(m,optimizer_constructor)
    optimize!(m);
    @test termination_status(m) == MOI.LOCALLY_SOLVED 
end


