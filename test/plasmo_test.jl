using Plasmo

atol = rtol = 1e-8
n_nodes=2
M = 100
d = sin.((0:M*n_nodes).*pi/100)
sol = [0.0,0.012006369082345895,0.03601910729568777,0.06464019379378924,0.09511095463076837,
       0.12658435685718497,0.15930888245402086,0.19490782553916158,0.23803327964088575,0.30104877201645763,
       0.41642314918303136,0.012006369082345895,-0.007398020864786417,-0.034169433031211886,-0.06363755248153519,
       -0.09385983133788765,-0.12370993944339496,-0.15178237150058388,-0.1750177872948184,-0.18567439478928288,
       -0.16361672887265558,0.41642314918303136,0.3602125748382587,0.35688741411044456,0.37370487528791935,
       0.39839711496710717,0.42693209836421125,0.4595099119322789,0.500790812935334,0.564583037956359,
       0.6876781516466413,0.9666692576858705,-0.056210574344772696,-0.03473591980594239,-0.045973058351838625,
       -0.06941607363932646,-0.09679825016720016,-0.12385665147216322,-0.14610041358266948,-0.1543510163755176,
       -0.12559477347457246,0.0]
    
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
    Dict(),
    Dict(:linear_solver=>"schur",  :schur_custom_partition=>true),
    Dict(:linear_solver=>"schwarz",:schwarz_custom_partition=>true)
]

n1 = getnode(graph,1)
@constraint(n1,n1[:x][1] == 0)

@testset "Plasmo test" for kwargs in kwargs_collection
    MadNLP.optimize!(graph;kwargs...);
    z = vcat([value.(all_variables(node)) for node in Plasmo.all_nodes(graph)]...)
    @test true
end
