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
        
@kwdef mutable struct RFSolverOptions <: MadNLP.AbstractOptions
    symbolic_analysis::Symbol = :klu
    fast_mode::Bool = true
    factorization_algo::CUSOLVER.cusolverRfFactorization_t = CUSOLVER.CUSOLVERRF_FACTORIZATION_ALG0
    triangular_solve_algo::CUSOLVER.cusolverRfTriangularSolve_t = CUSOLVER.CUSOLVERRF_TRIANGULAR_SOLVE_ALG1
end


const CuSubVector{T} = SubArray{T, 1, CUDA.CuArray{T, 1, CUDA.Mem.DeviceBuffer}, Tuple{CUDA.CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}}, false}

mutable struct RFSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing,CUSOLVERRF.RFLowLevel}

    tril::CUSPARSE.CuSparseMatrixCSC{T}
    full::CUSPARSE.CuSparseMatrixCSR{T}
    tril_to_full_view::CuSubVector{T}

    opt::RFSolverOptions
    logger::MadNLP.MadNLPLogger
end

function RFSolver(
    csc::CUSPARSE.CuSparseMatrixCSC{Float64};
    opt=RFSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
)
    n, m = size(csc)
    @assert n == m

    full,tril_to_full_view = MadNLP.get_tril_to_full(csc)
    
    return RFSolver{Float64}(
        nothing, csc, full, tril_to_full_view,
        opt, logger
    )
end

function MadNLP.factorize!(M::RFSolver)
    copyto!(M.full.nzVal, M.tril_to_full_view)
    if M.inner == nothing
        sym_lu = CUSOLVERRF.klu_symbolic_analysis(M.full)
        M.inner = CUSOLVERRF.RFLowLevel(
            sym_lu;
            fast_mode=M.opt.fast_mode,
            factorization_algo=M.opt.factorization_algo,
            triangular_algo=M.opt.triangular_solve_algo,
        )
    end
    CUSOLVERRF.rf_refactor!(M.inner, M.full)
    return M
end

function MadNLP.solve!(M::RFSolver{Float64}, x)
    CUSOLVERRF.rf_solve!(M.inner, x)
    return x
end

MadNLP.input_type(::Type{RFSolver}) = :csc
MadNLP.default_options(::Type{RFSolver}) = RFSolverOptions()
MadNLP.is_inertia(M::RFSolver) = false
MadNLP.improve!(M::RFSolver) = false
MadNLP.is_supported(::Type{RFSolver},::Type{Float32}) = false
MadNLP.is_supported(::Type{RFSolver},::Type{Float64}) = true
MadNLP.introduce(M::RFSolver) = "cuSolverRF"
