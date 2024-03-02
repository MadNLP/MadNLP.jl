using LinearAlgebra

@testset "[KKT vector] $KKTVector" for KKTVector in [
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

    # Test copy
    copy_rhs = copy(rhs)
    @test MadNLP.full(rhs) == MadNLP.full(copy_rhs)
end

@testset "[KKT system] $(KKTSystem)" for (KKTSystem, Callback) in [
    (MadNLP.SparseKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseUnreducedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseCondensedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.DenseKKTSystem, MadNLP.DenseCallback),
    (MadNLP.DenseCondensedKKTSystem, MadNLP.DenseCallback),
]
    linear_solver = MadNLP.LapackCPUSolver
    cnt = MadNLP.MadNLPCounters(; start_time=time())

    nlp = MadNLPTests.HS15Model()
    ind_cons = MadNLP.get_index_constraints(nlp)

    cb = MadNLP.create_callback(
        Callback, nlp,
    )

    kkt = MadNLP.create_kkt_system(
        KKTSystem,
        cb,
        ind_cons,
        linear_solver;
    )
    MadNLPTests.test_kkt_system(kkt, cb)
end

