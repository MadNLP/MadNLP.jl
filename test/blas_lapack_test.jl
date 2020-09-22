atol = rtol = 1e-8

m = 2
n = 2
row = Int32[1,2,2]
col = Int32[1,1,2]
val = Float64[1.,.1,2.]

A = sparse(row,col,val,m,n)
b = [1.0,3.0]
sol   = [1.0,6.1]
solt  = [1.3,6.0]
sysol = [1.3,6.1]

x = similar(sol)


@testset "BLAS" begin
    MadNLP.mv!(x,A,b)
    @test solcmp(x,sol,atol,rtol)

    MadNLP.mv!(x,A',b)
    @test solcmp(x,solt,atol,rtol)
    
    MadNLP.symv!(x,A,b)
    @test solcmp(x,sysol,atol,rtol)
end

A = Array(A)
sol= [0.8542713567839195, 1.4572864321608041]
    
@testset "LAPACK" begin
    M = MadNLP.LapackMKL.Solver(A)
    MadNLP.introduce(M)
    MadNLP.improve!(M)
    MadNLP.factorize!(M)
    MadNLP.is_inertia(M) && (MadNLP.inertia(M) = (2,0,0))
    x = MadNLP.solve!(M,copy(b))
    @test solcmp(x,sol,atol,rtol)
end
