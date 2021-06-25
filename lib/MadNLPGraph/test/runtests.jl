using Test, Plasmo, MadNLP, MadNLPTests, MadNLPGraph

@testset "MadNLPGraphs test" begin
    
    n_nodes=2
    M = 200
    d = sin.((0:M*n_nodes).*pi/100)
    
    graph = OptiGraph()
    @optinode(graph,nodes[1:n_nodes])

    #Node models
    nodecnt = 1
    for (i,node) in enumerate(nodes)
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
        ("default",Dict(:print_level=>MadNLP.ERROR)),
        ("schur",Dict(:linear_solver=>MadNLPSchur,  :schur_custom_partition=>true,:print_level=>MadNLP.ERROR)),
        ("schwarz",Dict(:linear_solver=>MadNLPSchwarz,:schwarz_custom_partition=>true,:print_level=>MadNLP.ERROR))
    ]
    for (name,kwargs) in kwargs_collection
        @testset "plasmo-$name" begin
            MadNLP.optimize!(graph;kwargs...);
            @test MadNLP.termination_status(graph.optimizer) == MOI.LOCALLY_SOLVED
        end
    end

    optimizer_constructors = [
        ("default",()->MadNLP.Optimizer(print_level=MadNLP.ERROR)),
        ("schur",()->MadNLP.Optimizer(linear_solver=MadNLPSchur,schur_num_parts=2,print_level=MadNLP.ERROR)),
        ("schwarz-single",()->MadNLP.Optimizer(linear_solver=MadNLPSchwarz,schwarz_num_parts=2,print_level=MadNLP.ERROR)),
        ("schwarz-double",()->MadNLP.Optimizer(linear_solver=MadNLPSchwarz,schwarz_num_parts_upper=2,schwarz_num_parts=10,
                                               print_level=MadNLP.ERROR))
    ]

    for (name,optimizer_constructor) in optimizer_constructors
        @testset "jump-$name" begin
            node,~=combine(graph)
            m = node.model
            set_optimizer(m,optimizer_constructor)
            optimize!(m);
            @test solcmp([0.0,0.03137979101284875,0.0627286139604959,0.09401553133948139,0.12520966673966746,0.15628023531552773,0.18719657416707308,0.21792817260043182,0.24844470223822107,0.2787160469499936], value.(getnode(graph,1)[:x][1:10]))
            @test termination_status(m) == MOI.LOCALLY_SOLVED
        end
    end
    
end
