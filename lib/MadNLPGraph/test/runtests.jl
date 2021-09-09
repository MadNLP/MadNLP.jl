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
            @test solcmp([0.0,0.03137979101284875,0.0627286139604959,0.09401553133948139,0.12520966673966746,0.15628023531552773,0.18719657416707308,0.21792817260043182,0.24844470223822107,0.2787160469499936], value.(getnode(graph,1)[:x][1:10]))
            @test solcmp([-0.0627595821831224,-0.06269764605256355,-0.06257383491492904,-0.06238827095686386,-0.06214113730759117,-0.06183267785818605,-0.06146319702088434,-0.06103305942866446,-0.06054268957539879,-0.05999257139692946], dual.(getnode(graph,1)[:dynamics][1:10]))
            @test solcmp([-1.574246826343827e-10,-1.5726901766520537e-10,-1.5695784406668639e-10,-1.5649147434189076e-10,-1.5587037679839438e-10,-1.5509517501550025e-10,-1.541666471342132e-10,-1.5308572497187857e-10,-1.5185349296508524e-10,-1.504711869423148e-10], dual.(UpperBoundRef.(getnode(graph,1)[:u][1:10])))
            @test solcmp([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dual.(LowerBoundRef.(getnode(graph,1)[:u][1:10])))
        end
    end
end
