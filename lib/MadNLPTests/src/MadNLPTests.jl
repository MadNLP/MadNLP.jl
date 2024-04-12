module MadNLPTests

# Standard packages
import LinearAlgebra: norm, I, mul!, dot
import SparseArrays: sparse
import Random
import Test: @test, @testset

# Optimization packages
import MadNLP
import NLPModels
import JuMP: Model, @variable, @NLconstraint, @NLobjective, @NLconstraint , @NLobjective, optimize!,
    MOI, termination_status, LowerBoundRef, UpperBoundRef, value, dual
import NLPModelsJuMP

export test_madnlp, solcmp

function solcmp(x,sol;atol=1e-4,rtol=1e-4)
    aerr = norm(x-sol,Inf)
    rerr = aerr/norm(sol,Inf)
    return (aerr < atol || rerr < rtol)
end

function test_linear_solver(solver, T; kwargs...)
    m = 2
    n = 2
    row = Int32[1,2,2]
    col = Int32[1,1,2]
    val = T[1.,.1,2.]
    b = T[1.0,3.0]
    x = similar(b)
    sol= [0.8542713567839195, 1.4572864321608041]

    csc = sparse(row,col,val,m,n)
    if MadNLP.input_type(solver) == :csc
        opt = MadNLP.default_options(solver)
        M = solver(csc; opt=opt)
    elseif MadNLP.input_type(solver) == :dense
        dense = Array(csc)
        opt = MadNLP.default_options(solver)
        M = solver(dense; opt=opt)
    end
    MadNLP.introduce(M)
    MadNLP.improve!(M)
    MadNLP.factorize!(M)
    if MadNLP.is_inertia(M)
        @test MadNLP.inertia(M) == (2, 0, 0)
    end
    x = MadNLP.solve!(M,copy(b))
    @test solcmp(x,sol)
end

function test_kkt_system(kkt, cb)
    # Getters
    n = MadNLP.num_variables(kkt)
    (m, p) = size(kkt)
    # system should be square
    @test m == p

    # Interface
    MadNLP.initialize!(kkt)

    # Update internal structure
    x0 = NLPModels.get_x0(cb.nlp)
    y0 = NLPModels.get_y0(cb.nlp)
    # Update Jacobian manually
    jac = MadNLP.get_jacobian(kkt)
    MadNLP._eval_jac_wrapper!(cb, x0, jac)
    MadNLP.compress_jacobian!(kkt)
    # Update Hessian manually
    hess = MadNLP.get_hessian(kkt)
    MadNLP._eval_lag_hess_wrapper!(cb, x0, y0, hess)
    MadNLP.compress_hessian!(kkt)

    # N.B.: set non-trivial dual's bounds to ensure
    # l_lower and u_lower are positive. If not we run into
    # an issue inside SparseUnreducedKKTSystem, which symmetrize
    # the system using the values in l_lower and u_lower.
    fill!(kkt.l_lower, 1e-3)
    fill!(kkt.u_lower, 1e-3)

    # Update diagonal terms manually.
    MadNLP._set_aug_diagonal!(kkt)

    # Factorization
    MadNLP.build_kkt!(kkt)
    MadNLP.factorize!(kkt.linear_solver)

    # Backsolve
    x = MadNLP.UnreducedKKTVector(kkt)
    fill!(MadNLP.full(x), 1.0)  # fill RHS with 1
    out1 = MadNLP.solve!(kkt, x)
    @test out1 === x

    y = copy(x)
    fill!(MadNLP.full(y), 0.0)
    out2 = mul!(y, kkt, x)
    @test out2 === y
    @test MadNLP.full(y) ≈ ones(length(x))

    if MadNLP.is_inertia(kkt.linear_solver)
        ni, mi, pi = MadNLP.inertia(kkt.linear_solver)
        @test MadNLP.is_inertia_correct(kkt, ni, mi, pi)
    end

    prim_reg, dual_reg = 1.0, 1.0
    MadNLP.regularize_diagonal!(kkt, prim_reg, dual_reg)

    return
end

function test_madnlp(name,optimizer_constructor::Function,exclude; Arr = Array)
    @testset "$name" begin
        for f in [infeasible,unbounded,lootsma,eigmina,lp]
            !(string(f) in exclude) && f(optimizer_constructor; Arr = Arr)
        end
    end
end

function infeasible(optimizer_constructor::Function; Arr = Array)
    @testset "infeasible" begin
        m=Model(optimizer_constructor)
        @variable(m,x>=1)
        @NLconstraint(m,x==0.)
        @NLobjective(m,Min,x^2)

        nlp = SparseWrapperModel(
            Arr,
            NLPModelsJuMP.MathOptNLPModel(m)
        )
        optimizer = optimizer_constructor()
        result = MadNLP.madnlp(nlp; optimizer.options...)

        @test result.status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
    end
end

function unbounded(optimizer_constructor::Function; Arr = Array)
    @testset "unbounded" begin
        m=Model(optimizer_constructor)
        @variable(m,x,start=1)
        @NLobjective(m,Max,x^2)

        nlp = SparseWrapperModel(
            Arr,
            NLPModelsJuMP.MathOptNLPModel(m)
        )
        optimizer = optimizer_constructor()
        result = MadNLP.madnlp(nlp; optimizer.options...)

        @test result.status == MadNLP.DIVERGING_ITERATES
    end
end

function lootsma(optimizer_constructor::Function; Arr = Array)
    @testset "lootsma" begin
        m=Model()
        @variable(m, par == 6.)
        @variable(m,0 <= x[i=1:3] <= 5, start = 0.)
        l=[
            @NLconstraint(m,-sqrt(x[1]) - sqrt(x[2]) + sqrt(x[3]) >= 0.)
            @NLconstraint(m,sqrt(x[1]) + sqrt(x[2]) + sqrt(x[3]) >= 4.)
        ]
        @NLobjective(m,Min,x[1]^3 + 11. *x[1] - par*sqrt(x[1])  +x[3] )


        nlp = SparseWrapperModel(
            Arr,
            NLPModelsJuMP.MathOptNLPModel(m)
        )

        optimizer = optimizer_constructor()
        result = MadNLP.madnlp(nlp; optimizer.options...)

        @test solcmp(
            Array(result.solution[2:4]),
            [0.07415998565403112,2.9848713863700236,4.0000304145340415];
            atol = sqrt(result.options.tol), rtol = sqrt(result.options.tol)
        )
        @test solcmp(
            Array(result.multipliers),
            [-2.000024518601535,-2.0000305441119535];
            atol = sqrt(result.options.tol), rtol = sqrt(result.options.tol)
        )
        @test solcmp(
            Array(result.multipliers_L[2:4]),
            [0.,0.,0.];
            atol = sqrt(result.options.tol), rtol = sqrt(result.options.tol)
        )
        @test solcmp(
            Array(result.multipliers_U[2:4]),
            [0.,0.,0.];
            atol = sqrt(result.options.tol), rtol = sqrt(result.options.tol)
        )

        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end
end

function eigmina(optimizer_constructor::Function; Arr = Array)
    @testset "eigmina" begin
        m=Model(optimizer_constructor)
        @variable(m,-1 <= x[1:101] <= 1,start = .1)
        @NLconstraint(m, x[1]*x[1] + x[2]*x[2] + x[3]*x[3] + x[4]*x[4] + x[5]*x[5] + x[6]*x[6] +
            x[7]*x[7] + x[8]*x[8] + x[9]*x[9] + x[10]*x[10] + x[11]*x[11] + x[12]*x[12] +
            x[13]*x[13] + x[14]*x[14] + x[15]*x[15] + x[16]*x[16] + x[17]*x[17] + x[18]*x[18] +
            x[19]*x[19] + x[20]*x[20] + x[21]*x[21] + x[22]*x[22] + x[23]*x[23] + x[24]*x[24] +
            x[25]*x[25] + x[26]*x[26] + x[27]*x[27] + x[28]*x[28] + x[29]*x[29] + x[30]*x[30] +
            x[31]*x[31] + x[32]*x[32] + x[33]*x[33] + x[34]*x[34] + x[35]*x[35] + x[36]*x[36] +
            x[37]*x[37] + x[38]*x[38] + x[39]*x[39] + x[40]*x[40] + x[41]*x[41] + x[42]*x[42] +
            x[43]*x[43] + x[44]*x[44] + x[45]*x[45] + x[46]*x[46] + x[47]*x[47] + x[48]*x[48] +
            x[49]*x[49] + x[50]*x[50] + x[51]*x[51] + x[52]*x[52] + x[53]*x[53] + x[54]*x[54] +
            x[55]*x[55] + x[56]*x[56] + x[57]*x[57] + x[58]*x[58] + x[59]*x[59] + x[60]*x[60] +
            x[61]*x[61] + x[62]*x[62] + x[63]*x[63] + x[64]*x[64] + x[65]*x[65] + x[66]*x[66] +
            x[67]*x[67] + x[68]*x[68] + x[69]*x[69] + x[70]*x[70] + x[71]*x[71] + x[72]*x[72] +
            x[73]*x[73] + x[74]*x[74] + x[75]*x[75] + x[76]*x[76] + x[77]*x[77] + x[78]*x[78] +
            x[79]*x[79] + x[80]*x[80] + x[81]*x[81] + x[82]*x[82] + x[83]*x[83] + x[84]*x[84] +
            x[85]*x[85] + x[86]*x[86] + x[87]*x[87] + x[88]*x[88] + x[89]*x[89] + x[90]*x[90] +
            x[91]*x[91] + x[92]*x[92] + x[93]*x[93] + x[94]*x[94] + x[95]*x[95] + x[96]*x[96] +
            x[97]*x[97] + x[98]*x[98] + x[99]*x[99] + x[100]*x[100] == 1)
        @NLconstraint(m, x[1]*x[101] - x[1] == 0)
        @NLconstraint(m, x[2]*x[101] - 2*x[2] == 0)
        @NLconstraint(m, x[3]*x[101] - 3*x[3] == 0)
        @NLconstraint(m, x[4]*x[101] - 4*x[4] == 0)
        @NLconstraint(m, x[5]*x[101] - 5*x[5] == 0)
        @NLconstraint(m, x[6]*x[101] - 6*x[6] == 0)
        @NLconstraint(m, x[7]*x[101] - 7*x[7] == 0)
        @NLconstraint(m, x[8]*x[101] - 8*x[8] == 0)
        @NLconstraint(m, x[9]*x[101] - 9*x[9] == 0)
        @NLconstraint(m, x[10]*x[101] - 10*x[10] == 0)
        @NLconstraint(m, x[11]*x[101] - 11*x[11] == 0)
        @NLconstraint(m, x[12]*x[101] - 12*x[12] == 0)
        @NLconstraint(m, x[13]*x[101] - 13*x[13] == 0)
        @NLconstraint(m, x[14]*x[101] - 14*x[14] == 0)
        @NLconstraint(m, x[15]*x[101] - 15*x[15] == 0)
        @NLconstraint(m, x[16]*x[101] - 16*x[16] == 0)
        @NLconstraint(m, x[17]*x[101] - 17*x[17] == 0)
        @NLconstraint(m, x[18]*x[101] - 18*x[18] == 0)
        @NLconstraint(m, x[19]*x[101] - 19*x[19] == 0)
        @NLconstraint(m, x[20]*x[101] - 20*x[20] == 0)
        @NLconstraint(m, x[21]*x[101] - 21*x[21] == 0)
        @NLconstraint(m, x[22]*x[101] - 22*x[22] == 0)
        @NLconstraint(m, x[23]*x[101] - 23*x[23] == 0)
        @NLconstraint(m, x[24]*x[101] - 24*x[24] == 0)
        @NLconstraint(m, x[25]*x[101] - 25*x[25] == 0)
        @NLconstraint(m, x[26]*x[101] - 26*x[26] == 0)
        @NLconstraint(m, x[27]*x[101] - 27*x[27] == 0)
        @NLconstraint(m, x[28]*x[101] - 28*x[28] == 0)
        @NLconstraint(m, x[29]*x[101] - 29*x[29] == 0)
        @NLconstraint(m, x[30]*x[101] - 30*x[30] == 0)
        @NLconstraint(m, x[31]*x[101] - 31*x[31] == 0)
        @NLconstraint(m, x[32]*x[101] - 32*x[32] == 0)
        @NLconstraint(m, x[33]*x[101] - 33*x[33] == 0)
        @NLconstraint(m, x[34]*x[101] - 34*x[34] == 0)
        @NLconstraint(m, x[35]*x[101] - 35*x[35] == 0)
        @NLconstraint(m, x[36]*x[101] - 36*x[36] == 0)
        @NLconstraint(m, x[37]*x[101] - 37*x[37] == 0)
        @NLconstraint(m, x[38]*x[101] - 38*x[38] == 0)
        @NLconstraint(m, x[39]*x[101] - 39*x[39] == 0)
        @NLconstraint(m, x[40]*x[101] - 40*x[40] == 0)
        @NLconstraint(m, x[41]*x[101] - 41*x[41] == 0)
        @NLconstraint(m, x[42]*x[101] - 42*x[42] == 0)
        @NLconstraint(m, x[43]*x[101] - 43*x[43] == 0)
        @NLconstraint(m, x[44]*x[101] - 44*x[44] == 0)
        @NLconstraint(m, x[45]*x[101] - 45*x[45] == 0)
        @NLconstraint(m, x[46]*x[101] - 46*x[46] == 0)
        @NLconstraint(m, x[47]*x[101] - 47*x[47] == 0)
        @NLconstraint(m, x[48]*x[101] - 48*x[48] == 0)
        @NLconstraint(m, x[49]*x[101] - 49*x[49] == 0)
        @NLconstraint(m, x[50]*x[101] - 50*x[50] == 0)
        @NLconstraint(m, x[51]*x[101] - 51*x[51] == 0)
        @NLconstraint(m, x[52]*x[101] - 52*x[52] == 0)
        @NLconstraint(m, x[53]*x[101] - 53*x[53] == 0)
        @NLconstraint(m, x[54]*x[101] - 54*x[54] == 0)
        @NLconstraint(m, x[55]*x[101] - 55*x[55] == 0)
        @NLconstraint(m, x[56]*x[101] - 56*x[56] == 0)
        @NLconstraint(m, x[57]*x[101] - 57*x[57] == 0)
        @NLconstraint(m, x[58]*x[101] - 58*x[58] == 0)
        @NLconstraint(m, x[59]*x[101] - 59*x[59] == 0)
        @NLconstraint(m, x[60]*x[101] - 60*x[60] == 0)
        @NLconstraint(m, x[61]*x[101] - 61*x[61] == 0)
        @NLconstraint(m, x[62]*x[101] - 62*x[62] == 0)
        @NLconstraint(m, x[63]*x[101] - 63*x[63] == 0)
        @NLconstraint(m, x[64]*x[101] - 64*x[64] == 0)
        @NLconstraint(m, x[65]*x[101] - 65*x[65] == 0)
        @NLconstraint(m, x[66]*x[101] - 66*x[66] == 0)
        @NLconstraint(m, x[67]*x[101] - 67*x[67] == 0)
        @NLconstraint(m, x[68]*x[101] - 68*x[68] == 0)
        @NLconstraint(m, x[69]*x[101] - 69*x[69] == 0)
        @NLconstraint(m, x[70]*x[101] - 70*x[70] == 0)
        @NLconstraint(m, x[71]*x[101] - 71*x[71] == 0)
        @NLconstraint(m, x[72]*x[101] - 72*x[72] == 0)
        @NLconstraint(m, x[73]*x[101] - 73*x[73] == 0)
        @NLconstraint(m, x[74]*x[101] - 74*x[74] == 0)
        @NLconstraint(m, x[75]*x[101] - 75*x[75] == 0)
        @NLconstraint(m, x[76]*x[101] - 76*x[76] == 0)
        @NLconstraint(m, x[77]*x[101] - 77*x[77] == 0)
        @NLconstraint(m, x[78]*x[101] - 78*x[78] == 0)
        @NLconstraint(m, x[79]*x[101] - 79*x[79] == 0)
        @NLconstraint(m, x[80]*x[101] - 80*x[80] == 0)
        @NLconstraint(m, x[81]*x[101] - 81*x[81] == 0)
        @NLconstraint(m, x[82]*x[101] - 82*x[82] == 0)
        @NLconstraint(m, x[83]*x[101] - 83*x[83] == 0)
        @NLconstraint(m, x[84]*x[101] - 84*x[84] == 0)
        @NLconstraint(m, x[85]*x[101] - 85*x[85] == 0)
        @NLconstraint(m, x[86]*x[101] - 86*x[86] == 0)
        @NLconstraint(m, x[87]*x[101] - 87*x[87] == 0)
        @NLconstraint(m, x[88]*x[101] - 88*x[88] == 0)
        @NLconstraint(m, x[89]*x[101] - 89*x[89] == 0)
        @NLconstraint(m, x[90]*x[101] - 90*x[90] == 0)
        @NLconstraint(m, x[91]*x[101] - 91*x[91] == 0)
        @NLconstraint(m, x[92]*x[101] - 92*x[92] == 0)
        @NLconstraint(m, x[93]*x[101] - 93*x[93] == 0)
        @NLconstraint(m, x[94]*x[101] - 94*x[94] == 0)
        @NLconstraint(m, x[95]*x[101] - 95*x[95] == 0)
        @NLconstraint(m, x[96]*x[101] - 96*x[96] == 0)
        @NLconstraint(m, x[97]*x[101] - 97*x[97] == 0)
        @NLconstraint(m, x[98]*x[101] - 98*x[98] == 0)
        @NLconstraint(m, x[99]*x[101] - 99*x[99] == 0)
        @NLconstraint(m, x[100]*x[101] - 100*x[100] == 0)
        @NLobjective(m, Min, x[101])

        nlp = SparseWrapperModel(
            Arr,
            NLPModelsJuMP.MathOptNLPModel(m)
        )
        optimizer = optimizer_constructor()
        result = MadNLP.madnlp(nlp; optimizer.options...)

        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end
end

function lp(optimizer_constructor::Function; Arr = Array)
    @testset "lp" begin
        
        model = Model()
        @variable(model, x >= 0)
        @variable(model, 0 <= y <= 3)
        @objective(model, Min, 12x + 20y)
        @constraint(model, c1, 6x + 8y >= 100)
        @constraint(model, c2, 7x + 12y >= 120)

        nlp = SparseWrapperModel(
            Arr,
            NLPModelsJuMP.MathOptNLPModel(m)
        )
        optimizer = optimizer_constructor()
        result = MadNLP.madnlp(nlp; optimizer.options...)

        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end
end

include("Instances/dummy_qp.jl")
include("Instances/hs15.jl")
include("Instances/nls.jl")
include("wrapper.jl")

end # module
