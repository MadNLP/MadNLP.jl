module MadNLPTests

# Standard packages
import LinearAlgebra: norm, I, mul!, dot
import Random
import Test: @test, @testset

# Optimization packages
import MadNLP
import NLPModels
import JuMP: Model, @variable, @constraint, @objective, @NLconstraint , @NLobjective, optimize!,
    MOI, termination_status, LowerBoundRef, UpperBoundRef, value, dual

export test_madnlp, solcmp

function solcmp(x,sol;atol=1e-4,rtol=1e-4)
    aerr = norm(x-sol,Inf)
    rerr = aerr/norm(sol,Inf)
    return (aerr < atol || rerr < rtol)
end

function test_madnlp(name,optimizer_constructor::Function,exclude)
    @testset "$name" begin
        for f in [infeasible,unbounded,lootsma,eigmina]
            !(string(f) in exclude) && f(optimizer_constructor)
        end
    end
end

function infeasible(optimizer_constructor::Function)
    @testset "infeasible" begin
        m=Model(optimizer_constructor)
        @variable(m,x>=1)
        @constraint(m,x==0.)
        @objective(m,Min,x^2)
        optimize!(m)
        @test termination_status(m) == MOI.LOCALLY_INFEASIBLE
    end
end

function unbounded(optimizer_constructor::Function)
    @testset "unbounded" begin
        m=Model(optimizer_constructor)
        @variable(m,x,start=1)
        @objective(m,Max,x^2)
        optimize!(m)
        @test termination_status(m) == MOI.INFEASIBLE_OR_UNBOUNDED
    end
end

function lootsma(optimizer_constructor::Function)
    @testset "lootsma" begin
        m=Model(optimizer_constructor)
        @variable(m, par == 6.)
        @variable(m,0 <= x[i=1:3] <= 5, start = 0.)
        l=[
            @NLconstraint(m,-sqrt(x[1]) - sqrt(x[2]) + sqrt(x[3]) >= 0.)
            @NLconstraint(m,sqrt(x[1]) + sqrt(x[2]) + sqrt(x[3]) >= 4.)
        ]
        @NLobjective(m,Min,x[1]^3 + 11. *x[1] - par*sqrt(x[1])  +x[3] )

        optimize!(m)

        @test solcmp(value.(x),[0.07415998565403112,2.9848713863700236,4.0000304145340415])
        @test solcmp(dual.(l),[2.000024518601535,2.0000305441119535])
        @test solcmp(dual.(LowerBoundRef.(x)),[0.,0.,0.])
        @test solcmp(dual.(UpperBoundRef.(x)),[0.,0.,0.])

        @test termination_status(m) == MOI.LOCALLY_SOLVED
    end
end

function eigmina(optimizer_constructor::Function)
    @testset "eigmina" begin
        m=Model(optimizer_constructor)
        @variable(m,-1 <= x[1:101] <= 1,start = .1)
        @constraint(m, x[1]*x[1] + x[2]*x[2] + x[3]*x[3] + x[4]*x[4] + x[5]*x[5] + x[6]*x[6] +
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
        @constraint(m, x[1]*x[101] - x[1] == 0)
        @constraint(m, x[2]*x[101] - 2*x[2] == 0)
        @constraint(m, x[3]*x[101] - 3*x[3] == 0)
        @constraint(m, x[4]*x[101] - 4*x[4] == 0)
        @constraint(m, x[5]*x[101] - 5*x[5] == 0)
        @constraint(m, x[6]*x[101] - 6*x[6] == 0)
        @constraint(m, x[7]*x[101] - 7*x[7] == 0)
        @constraint(m, x[8]*x[101] - 8*x[8] == 0)
        @constraint(m, x[9]*x[101] - 9*x[9] == 0)
        @constraint(m, x[10]*x[101] - 10*x[10] == 0)
        @constraint(m, x[11]*x[101] - 11*x[11] == 0)
        @constraint(m, x[12]*x[101] - 12*x[12] == 0)
        @constraint(m, x[13]*x[101] - 13*x[13] == 0)
        @constraint(m, x[14]*x[101] - 14*x[14] == 0)
        @constraint(m, x[15]*x[101] - 15*x[15] == 0)
        @constraint(m, x[16]*x[101] - 16*x[16] == 0)
        @constraint(m, x[17]*x[101] - 17*x[17] == 0)
        @constraint(m, x[18]*x[101] - 18*x[18] == 0)
        @constraint(m, x[19]*x[101] - 19*x[19] == 0)
        @constraint(m, x[20]*x[101] - 20*x[20] == 0)
        @constraint(m, x[21]*x[101] - 21*x[21] == 0)
        @constraint(m, x[22]*x[101] - 22*x[22] == 0)
        @constraint(m, x[23]*x[101] - 23*x[23] == 0)
        @constraint(m, x[24]*x[101] - 24*x[24] == 0)
        @constraint(m, x[25]*x[101] - 25*x[25] == 0)
        @constraint(m, x[26]*x[101] - 26*x[26] == 0)
        @constraint(m, x[27]*x[101] - 27*x[27] == 0)
        @constraint(m, x[28]*x[101] - 28*x[28] == 0)
        @constraint(m, x[29]*x[101] - 29*x[29] == 0)
        @constraint(m, x[30]*x[101] - 30*x[30] == 0)
        @constraint(m, x[31]*x[101] - 31*x[31] == 0)
        @constraint(m, x[32]*x[101] - 32*x[32] == 0)
        @constraint(m, x[33]*x[101] - 33*x[33] == 0)
        @constraint(m, x[34]*x[101] - 34*x[34] == 0)
        @constraint(m, x[35]*x[101] - 35*x[35] == 0)
        @constraint(m, x[36]*x[101] - 36*x[36] == 0)
        @constraint(m, x[37]*x[101] - 37*x[37] == 0)
        @constraint(m, x[38]*x[101] - 38*x[38] == 0)
        @constraint(m, x[39]*x[101] - 39*x[39] == 0)
        @constraint(m, x[40]*x[101] - 40*x[40] == 0)
        @constraint(m, x[41]*x[101] - 41*x[41] == 0)
        @constraint(m, x[42]*x[101] - 42*x[42] == 0)
        @constraint(m, x[43]*x[101] - 43*x[43] == 0)
        @constraint(m, x[44]*x[101] - 44*x[44] == 0)
        @constraint(m, x[45]*x[101] - 45*x[45] == 0)
        @constraint(m, x[46]*x[101] - 46*x[46] == 0)
        @constraint(m, x[47]*x[101] - 47*x[47] == 0)
        @constraint(m, x[48]*x[101] - 48*x[48] == 0)
        @constraint(m, x[49]*x[101] - 49*x[49] == 0)
        @constraint(m, x[50]*x[101] - 50*x[50] == 0)
        @constraint(m, x[51]*x[101] - 51*x[51] == 0)
        @constraint(m, x[52]*x[101] - 52*x[52] == 0)
        @constraint(m, x[53]*x[101] - 53*x[53] == 0)
        @constraint(m, x[54]*x[101] - 54*x[54] == 0)
        @constraint(m, x[55]*x[101] - 55*x[55] == 0)
        @constraint(m, x[56]*x[101] - 56*x[56] == 0)
        @constraint(m, x[57]*x[101] - 57*x[57] == 0)
        @constraint(m, x[58]*x[101] - 58*x[58] == 0)
        @constraint(m, x[59]*x[101] - 59*x[59] == 0)
        @constraint(m, x[60]*x[101] - 60*x[60] == 0)
        @constraint(m, x[61]*x[101] - 61*x[61] == 0)
        @constraint(m, x[62]*x[101] - 62*x[62] == 0)
        @constraint(m, x[63]*x[101] - 63*x[63] == 0)
        @constraint(m, x[64]*x[101] - 64*x[64] == 0)
        @constraint(m, x[65]*x[101] - 65*x[65] == 0)
        @constraint(m, x[66]*x[101] - 66*x[66] == 0)
        @constraint(m, x[67]*x[101] - 67*x[67] == 0)
        @constraint(m, x[68]*x[101] - 68*x[68] == 0)
        @constraint(m, x[69]*x[101] - 69*x[69] == 0)
        @constraint(m, x[70]*x[101] - 70*x[70] == 0)
        @constraint(m, x[71]*x[101] - 71*x[71] == 0)
        @constraint(m, x[72]*x[101] - 72*x[72] == 0)
        @constraint(m, x[73]*x[101] - 73*x[73] == 0)
        @constraint(m, x[74]*x[101] - 74*x[74] == 0)
        @constraint(m, x[75]*x[101] - 75*x[75] == 0)
        @constraint(m, x[76]*x[101] - 76*x[76] == 0)
        @constraint(m, x[77]*x[101] - 77*x[77] == 0)
        @constraint(m, x[78]*x[101] - 78*x[78] == 0)
        @constraint(m, x[79]*x[101] - 79*x[79] == 0)
        @constraint(m, x[80]*x[101] - 80*x[80] == 0)
        @constraint(m, x[81]*x[101] - 81*x[81] == 0)
        @constraint(m, x[82]*x[101] - 82*x[82] == 0)
        @constraint(m, x[83]*x[101] - 83*x[83] == 0)
        @constraint(m, x[84]*x[101] - 84*x[84] == 0)
        @constraint(m, x[85]*x[101] - 85*x[85] == 0)
        @constraint(m, x[86]*x[101] - 86*x[86] == 0)
        @constraint(m, x[87]*x[101] - 87*x[87] == 0)
        @constraint(m, x[88]*x[101] - 88*x[88] == 0)
        @constraint(m, x[89]*x[101] - 89*x[89] == 0)
        @constraint(m, x[90]*x[101] - 90*x[90] == 0)
        @constraint(m, x[91]*x[101] - 91*x[91] == 0)
        @constraint(m, x[92]*x[101] - 92*x[92] == 0)
        @constraint(m, x[93]*x[101] - 93*x[93] == 0)
        @constraint(m, x[94]*x[101] - 94*x[94] == 0)
        @constraint(m, x[95]*x[101] - 95*x[95] == 0)
        @constraint(m, x[96]*x[101] - 96*x[96] == 0)
        @constraint(m, x[97]*x[101] - 97*x[97] == 0)
        @constraint(m, x[98]*x[101] - 98*x[98] == 0)
        @constraint(m, x[99]*x[101] - 99*x[99] == 0)
        @constraint(m, x[100]*x[101] - 100*x[100] == 0)
        @objective(m, Min, x[101])
        optimize!(m)

        @test termination_status(m) == MOI.LOCALLY_SOLVED
    end
end


struct DenseDummyQP <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    P::Matrix{Float64} # primal hessian
    A::Matrix{Float64} # constraint jacobian
    q::Vector{Float64}
    hrows::Vector{Int}
    hcols::Vector{Int}
    jrows::Vector{Int}
    jcols::Vector{Int}
    counters::NLPModels.Counters
end

function NLPModels.jac_structure!(qp::DenseDummyQP, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, qp.jrows)
    copyto!(J, qp.jcols)
end
function NLPModels.hess_structure!(qp::DenseDummyQP, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, qp.hrows)
    copyto!(J, qp.hcols)
end

function NLPModels.obj(qp::DenseDummyQP, x::AbstractVector)
    return 0.5 * dot(x, qp.P, x) + dot(qp.q, x)
end
function NLPModels.grad!(qp::DenseDummyQP, x::AbstractVector, g::AbstractVector)
    mul!(g, qp.P, x)
    g .+= qp.q
    return
end
function NLPModels.cons!(qp::DenseDummyQP, x::AbstractVector, c::AbstractVector)
    mul!(c, qp.A, x)
end
# Jacobian: sparse callback
function NLPModels.jac_coord!(qp::DenseDummyQP, x::AbstractVector, J::AbstractVector)
    index = 1
    for (i, j) in zip(qp.jrows, qp.jcols)
        J[index] = qp.A[i, j]
        index += 1
    end
end
# Jacobian: dense callback
MadNLP.jac_dense!(qp::DenseDummyQP, x, J::AbstractMatrix) = copyto!(J, qp.A)
# Hessian: sparse callback
function NLPModels.hess_coord!(qp::DenseDummyQP,x, l, hess::AbstractVector; obj_weight=1.0)
    index = 1
    for i in 1:NLPModels.get_nvar(qp) , j in 1:i
        hess[index] = obj_weight * qp.P[j, i]
        index += 1
    end
end
# Hessian: dense callback
function MadNLP.hess_dense!(qp::DenseDummyQP, x, l,hess::AbstractMatrix; obj_weight=1.0)
    copyto!(hess, obj_weight .* qp.P)
end

function DenseDummyQP(; n=100, m=10, fixed_variables=Int[])
    if m >= n
        error("The number of constraints `m` should be less than the number of variable `n`.")
    end

    Random.seed!(1)

    # Build QP problem 0.5 * x' * P * x + q' * x
    P = randn(n , n)
    P += P' # P is symmetric
    P += 100.0 * I

    q = randn(n)

    # Build constraints gl <= Ax <= gu
    A = zeros(m, n)
    for j in 1:m
        A[j, j]  = 1.0
        A[j, j+1]  = -1.0
    end

    x0 = zeros(n)
    y0 = zeros(m)

    # Bound constraints
    xu =   ones(n)
    xl = - ones(n)
    gl = -ones(m)
    gu = ones(m)

    xl[fixed_variables] .= xu[fixed_variables]

    hrows = [i for i in 1:n for j in 1:i]
    hcols = [j for i in 1:n for j in 1:i]
    nnzh = div(n * (n + 1), 2)

    jrows = [j for i in 1:n for j in 1:m]
    jcols = [i for i in 1:n for j in 1:m]
    nnzj = n * m

    return DenseDummyQP(
        NLPModels.NLPModelMeta(
            n,
            ncon = m,
            nnzj = nnzj,
            nnzh = nnzh,
            x0 = x0,
            y0 = y0,
            lvar = xl,
            uvar = xu,
            lcon = gl,
            ucon = gu,
            minimize = true
        ),
        P,A,q,hrows,hcols,jrows,jcols,
        NLPModels.Counters()
    )
end

end # module
