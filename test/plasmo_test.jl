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
for i=1:n_nodes-1
    @linkconstraint(graph, nodes[i+1][:x][1] == nodes[i][:x][M] + nodes[i][:u][M],attach=nodes[i+1])
end

#First node gets initial condition
kwargs_collection = [
    Dict(:log_level=>"error"),
    Dict(:linear_solver=>"schur",  :schur_custom_partition=>true,:log_level=>"error"),
    Dict(:linear_solver=>"schwarz",:schwarz_custom_partition=>true,:log_level=>"error")
]

n1 = getnode(graph,1)
@constraint(n1,n1[:x][1] == 0)

@testset "Plasmo test" for kwargs in kwargs_collection
    MadNLP.optimize!(graph;kwargs...);
    @test termination_status(graph.optimizer) == MadNLP.MOI.LOCALLY_SOLVED 
    # optimize!(graph,()->MadNLP.Optimizer(kwargs...));
    # @test termination_status(graph.optimizer) == MadNLP.MOI.LOCALLY_SOLVED 
end
