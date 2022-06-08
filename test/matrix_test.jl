atol = rtol = 1e-4

m = 2
n = 2
row = Int32[1,2,2]
col = Int32[1,1,2]
val = Float64[1.,.1,2.]

coo = MadNLP.SparseMatrixCOO(m,n,row,col,val)
csc = sparse(row,col,val,m,n)
dense = Array(csc)

b = [1.0,3.0]
x = similar(b)

@testset "SparseMatrixCOO" begin
    @test MadNLP.size(coo) == (2,2)
    @test coo[1,1] == 1.
end

@testset "LAPACK" begin
    sol= [0.8542713567839195, 1.4572864321608041]
    M = MadNLP.LapackCPUSolver(dense)
    MadNLP.introduce(M)
    MadNLP.improve!(M)
    MadNLP.factorize!(M)
    MadNLP.is_inertia(M) && (MadNLP.inertia(M) = (2,0,0))
    x = MadNLP.solve!(M,copy(b))
    @test solcmp(x,sol)
end


MadNLPTests.test_linear_solver(MadNLP.LapackCPUSolver)
MadNLPTests.test_linear_solver(MadNLP.UmfpackSolver)
# @test_linear_solver ma27
# @test_linear_solver ma57
# @test_linear_solver ma77
# @test_linear_solver ma86
# @test_linear_solver ma97
# @test_linear_solver pardiso
# @test_linear_solver pardisomkl
# @test_linear_solver umfpack
# @test_linear_solver mumps
