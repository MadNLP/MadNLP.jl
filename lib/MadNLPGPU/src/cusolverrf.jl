# MIT License

# Copyright (c) 2020 Exanauts

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
const CuSubVector{T} = SubArray{T, 1, CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}, Tuple{CUDA.CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}}, false}

#=
    cusolverRF
=#

@kwdef mutable struct RFSolverOptions <: MadNLP.AbstractOptions
    rf_symbolic_analysis::Symbol = :klu
    rf_fast_mode::Bool = true
    rf_pivot_tol::Float64 = 1e-14
    rf_boost::Float64 = 1e-14
    rf_factorization_algo::CUSOLVER.cusolverRfFactorization_t = CUSOLVER.CUSOLVERRF_FACTORIZATION_ALG0
    rf_triangular_solve_algo::CUSOLVER.cusolverRfTriangularSolve_t = CUSOLVER.CUSOLVERRF_TRIANGULAR_SOLVE_ALG1
end

mutable struct RFSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing,CUSOLVERRF.RFLowLevel}

    tril::CUSPARSE.CuSparseMatrixCSC{T}
    full::CUSPARSE.CuSparseMatrixCSR{T}
    tril_to_full_view::CuSubVector{T}
    buffer::CUDA.CuVector{T}

    opt::RFSolverOptions
    logger::MadNLP.MadNLPLogger
end

function RFSolver(
    csc::CUSPARSE.CuSparseMatrixCSC;
    opt=RFSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
)
    n, m = size(csc)
    @assert n == m

    full,tril_to_full_view = MadNLP.get_tril_to_full(csc)

    full = CUSPARSE.CuSparseMatrixCSR(
        full.colPtr,
        full.rowVal,
        full.nzVal,
        full.dims
    )

    return RFSolver(
        nothing, csc, full, tril_to_full_view, similar(csc.nzVal,1),
        opt, logger
    )
end

function MadNLP.factorize!(M::RFSolver)
    copyto!(M.full.nzVal, M.tril_to_full_view)
    if M.inner == nothing
        sym_lu = CUSOLVERRF.klu_symbolic_analysis(M.full)
        M.inner = CUSOLVERRF.RFLowLevel(
            sym_lu;
            fast_mode=M.opt.rf_fast_mode,
            factorization_algo=M.opt.rf_factorization_algo,
            triangular_algo=M.opt.rf_triangular_solve_algo,
            # nboost=M.opt.rf_boost,
            # nzero=M.opt.rf_pivot_tol,
        )
    end
    CUSOLVERRF.rf_refactor!(M.inner, M.full)
    return M
end

function MadNLP.solve!(M::RFSolver{T}, x) where T
    CUSOLVERRF.rf_solve!(M.inner, x)
    # this is necessary to not distort the timing in MadNLP
    copyto!(M.buffer, M.buffer)
    synchronize(CUDABackend())
    # -----------------------------------------------------
    return x
end

MadNLP.input_type(::Type{RFSolver}) = :csc
MadNLP.default_options(::Type{RFSolver}) = RFSolverOptions()
MadNLP.is_inertia(M::RFSolver) = false
MadNLP.improve!(M::RFSolver) = false
MadNLP.is_supported(::Type{RFSolver},::Type{Float32}) = true
MadNLP.is_supported(::Type{RFSolver},::Type{Float64}) = true
MadNLP.introduce(M::RFSolver) = "cuSolverRF"


#=
    GLU
=#

@kwdef mutable struct GLUSolverOptions <: MadNLP.AbstractOptions
    glu_symbolic_analysis::Symbol = :klu
end

mutable struct GLUSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing,CUSOLVERRF.GLULowLevel}

    tril::CUSPARSE.CuSparseMatrixCSC{T}
    full::CUSPARSE.CuSparseMatrixCSR{T}
    tril_to_full_view::CuSubVector{T}
    buffer::CUDA.CuVector{T}

    opt::GLUSolverOptions
    logger::MadNLP.MadNLPLogger
end

function GLUSolver(
    csc::CUSPARSE.CuSparseMatrixCSC;
    opt=GLUSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
)
    n, m = size(csc)
    @assert n == m

    full,tril_to_full_view = MadNLP.get_tril_to_full(csc)

    full = CUSPARSE.CuSparseMatrixCSR(
        full.colPtr,
        full.rowVal,
        full.nzVal,
        full.dims
    )

    return GLUSolver(
        nothing, csc, full, tril_to_full_view, similar(csc.nzVal,1),
        opt, logger
    )
end

function MadNLP.factorize!(M::GLUSolver)
    copyto!(M.full.nzVal, M.tril_to_full_view)
    if M.inner == nothing  
        sym_lu = CUSOLVERRF.klu_symbolic_analysis(M.full)
        M.inner = CUSOLVERRF.GLULowLevel(sym_lu)
    end
    CUSOLVERRF.glu_refactor!(M.inner, M.full)
    return M
end

function MadNLP.solve!(M::GLUSolver{T}, x) where T
    CUSOLVERRF.glu_solve!(M.inner, x)
    # this is necessary to not distort the timing in MadNLP
    copyto!(M.buffer, M.buffer)
    synchronize(CUDABackend())
    # -----------------------------------------------------
    return x
end

MadNLP.input_type(::Type{GLUSolver}) = :csc
MadNLP.default_options(::Type{GLUSolver}) = GLUSolverOptions()
MadNLP.is_inertia(M::GLUSolver) = false
MadNLP.improve!(M::GLUSolver) = false
MadNLP.is_supported(::Type{GLUSolver},::Type{Float32}) = true
MadNLP.is_supported(::Type{GLUSolver},::Type{Float64}) = true
MadNLP.introduce(M::GLUSolver) = "GLU"



#=
Cholesky
=#
@kwdef mutable struct CuCholeskySolverOptions <: MadNLP.AbstractOptions
    # rf_pivot_tol::Float64 = 1e-14
    # rf_boost::Float64 = 1e-14
    # rf_factorization_algo::CUSOLVER.cusolverRfFactorization_t = CUSOLVER.CUSOLVERCholesky_FACTORIZATION_ALG0
    # rf_triangular_solve_algo::CUSOLVER.cusolverRfTriangularSolve_t = CUSOLVER.CUSOLVERCholesky_TRIANGULAR_SOLVE_ALG1
end

mutable struct CuCholeskySolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing,CUSOLVER.SparseCholesky}

    tril::CUSPARSE.CuSparseMatrixCSC{T}
    full::CUSPARSE.CuSparseMatrixCSR{T}
    tril_to_full_view::CuSubVector{T}
    buffer::CUDA.CuVector{T}

    fullp::CUSPARSE.CuSparseMatrixCSR{T}
    p::CUDA.CuVector{Int}
    pnzval::CUDA.CuVector{Int}
    rhs::CUDA.CuVector{T}
    
    singularity::Bool
    
    opt::CuCholeskySolverOptions
    logger::MadNLP.MadNLPLogger
end

function CuCholeskySolver(
    csc::CUSPARSE.CuSparseMatrixCSC;
    opt=CuCholeskySolverOptions(),
    logger=MadNLP.MadNLPLogger(),
)
    n, m = size(csc)
    @assert n == m

    full,tril_to_full_view = MadNLP.get_tril_to_full(csc)
    buffer = similar(csc.nzVal,1)

    full = CUSPARSE.CuSparseMatrixCSR(
        full.colPtr,
        full.rowVal,
        full.nzVal,
        full.dims
    )

    a = MadNLP.SparseMatrixCSC(full)
    aa = MadNLP.SparseMatrixCSC(
        n, m,
        a.colptr,
        a.rowval,
        Array(1:length(a.nzval)),
    )
    p = AMD.amd(aa)
    b = aa[p,p]
    pnzval = b.nzval    
    fullp = CUSPARSE.CuSparseMatrixCSR(
        CuArray(b.colptr),
        CuArray(b.rowval),
        similar(full.nzVal),
        (n, m),
    )    
    
    rhs = similar(csc.nzVal, n)

    return CuCholeskySolver(
        nothing, csc, full, tril_to_full_view, buffer,
        fullp, CuArray{Int}(p), CuArray{Int}(pnzval), rhs, false,
        opt, logger
    )
end

function MadNLP.factorize!(M::CuCholeskySolver)

    # println("B")
    # @time begin
        copyto!(M.full.nzVal, M.tril_to_full_view)
        _copy_to_perm_2!(CUDABackend())(M.fullp.nzVal, M.pnzval, M.full.nzVal; ndrange= length(M.pnzval))
        synchronize(CUDABackend())
        if M.inner == nothing
            M.inner = CUSOLVER.SparseCholesky(M.fullp)
            CUSOLVER.spcholesky_buffer(M.inner, M.fullp)
        end
        try
            CUSOLVER.spcholesky_factorise(M.inner, M.fullp, eltype(M.fullp.nzVal) == Float32 ? 1e-6 : 1e-12)
            M.singularity = false
        catch e
            println(e)
            M.singularity = true
        end
    # end
    return M
end

function MadNLP.solve!(M::CuCholeskySolver{T}, x) where T
    # println("A")
    # @time begin
        _copy_to_perm_2!(CUDABackend())(M.rhs, M.p, x; ndrange=length(M.p))
        synchronize(CUDABackend())
    # end
    # @time begin
        CUSOLVER.spcholesky_solve(M.inner, M.rhs, x)
    # end
    # @time begin
        _copy_to_perm!(CUDABackend())(M.rhs, M.p, x; ndrange=length(M.p))
        synchronize(CUDABackend())
        copyto!(x, M.rhs)
    # end
    return x
end

@kernel function _copy_to_perm!(y,p,x)
    i = @index(Global)
    @inbounds y[p[i]] = x[i]
end
@kernel function _copy_to_perm_2!(y,p,x)
    i = @index(Global)
    @inbounds y[i] = x[p[i]]
end
function MadNLP.inertia(M::CuCholeskySolver{T}) where T
    return !(M.singularity) ? (size(M.fullp,1),0,0) : (0,1,1)
end

MadNLP.input_type(::Type{CuCholeskySolver}) = :csc
MadNLP.default_options(::Type{CuCholeskySolver}) = CuCholeskySolverOptions()
MadNLP.is_inertia(M::CuCholeskySolver) = true
MadNLP.improve!(M::CuCholeskySolver) = false
MadNLP.is_supported(::Type{CuCholeskySolver},::Type{Float32}) = true
MadNLP.is_supported(::Type{CuCholeskySolver},::Type{Float64}) = true
MadNLP.introduce(M::CuCholeskySolver) = "cuSolverCholesky"

export CuCholeskySolver
