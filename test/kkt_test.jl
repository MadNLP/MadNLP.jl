using LinearAlgebra

@testset "$KKTVector" for KKTVector in [
    MadNLP.UnreducedKKTVector,
]
    T = Float64
    VT = Vector{Float64}
    n, m = 10, 20
    nlb, nub = 5, 6

    ind_lb = [2,3,4]
    ind_ub = [4,5,6]

    rhs = KKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    @test length(rhs) == length(MadNLP.full(rhs))
    @test MadNLP.number_primal(rhs) == length(MadNLP.primal(rhs)) == n
    @test MadNLP.number_dual(rhs) == length(MadNLP.dual(rhs)) == m
    @test norm(rhs) == 0

    fill!(rhs, one(T))
    @test norm(rhs) == sqrt(length(rhs))
end
