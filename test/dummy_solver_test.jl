# DummyLinearSolver is MadNLPCore's no-op default sparse solver (MadNLP overrides
# it to MumpsSolver), so it is otherwise never exercised. Cover its interface.
@testset "DummyLinearSolver" begin
    A = sparse([1.0 0.0; 0.0 2.0])
    M = MadNLP.DummyLinearSolver(A)
    @test MadNLP.input_type(MadNLP.DummyLinearSolver) == :csc
    @test MadNLP.is_supported(MadNLP.DummyLinearSolver, Float64)
    @test occursin("Dummy", MadNLP.introduce(M))
    MadNLP.factorize!(M)
    b = [1.0, 3.0]
    @test MadNLP.solve_linear_system!(M, b) === b   # no-op: rhs returned unchanged
    @test MadNLP.is_inertia(M) == false
    @test MadNLP.improve!(M) == false
end
