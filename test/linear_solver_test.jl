using SparseArrays, LinearAlgebra

A  = sparse(Int32[1,2,2],Int32[1,1,2],Float64[1.,.1,2.],2,2)
b  = [1.0,3.0]
sol= [0.8542713567839195, 1.4572864321608041]
atol = 1e-4; rtol = 1e-4

macro test_linear_solver(name)
    str=string(name)
    quote
        if isdefined(MadNLP,Symbol($str))
            @testset $str begin
                M = MadNLP.$name.Solver(A)
                MadNLP.introduce(M)
                MadNLP.improve!(M)
                MadNLP.factorize!(M)
                MadNLP.is_inertia(M) && (MadNLP.inertia(M) = (2,0,0))
                x = MadNLP.solve!(M,copy(b))
                @test solcmp(x,sol,atol,rtol)
            end
        end
    end
end

@test_linear_solver ma27
@test_linear_solver ma57
@test_linear_solver ma77
@test_linear_solver ma86
@test_linear_solver ma97
@test_linear_solver pardiso
@test_linear_solver pardisomkl
@test_linear_solver umfpack
@test_linear_solver mumps
