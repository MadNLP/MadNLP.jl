module MadNLPSchur

import ..MadNLPGraph:
    @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, EmptyLinearSolver, set_options!, SparseMatrixCSC, SubVector, 
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia,
    default_linear_solver, default_dense_solver, get_csc_view, get_cscsy_view, nnz, mul!,
    TwoStagePartition, set_blas_num_threads, blas_num_threads, @blas_safe_threads

const INPUT_MATRIX_TYPE = :csc


@kwdef mutable struct Options <: AbstractOptions
    schur_custom_partition::Bool = false
    schur_num_parts::Int = 2
    schur_part::Vector{Int} = Int[]
    schur_subproblem_solver::Module
    schur_dense_solver::Module
end

mutable struct SolverWorker
    V::Vector{Int}
    V_0_nz::Vector{Int}
    csc::SparseMatrixCSC{Float64,Int32}
    csc_view::SubVector{Float64}
    compl::SparseMatrixCSC{Float64,Int32}
    compl_view::SubVector{Float64}
    M::AbstractLinearSolver
    w::Vector{Float64}
end

mutable struct Solver <: AbstractLinearSolver
    csc::SparseMatrixCSC{Float64,Int32}
    inds::Vector{Int}
    tsp::TwoStagePartition

    schur::Matrix{Float64}
    colors
    fact

    V_0::Vector{Int}
    csc_0::SparseMatrixCSC{Float64,Int32}
    csc_0_view::SubVector{Float64}
    w_0::Vector{Float64}

    sws::Vector{SolverWorker}

    opt::Options
    logger::Logger
end


function Solver(csc::SparseMatrixCSC{Float64};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(schur_subproblem_solver=default_linear_solver(),
                            schur_dense_solver=default_dense_solver()),
                logger=Logger(),
                kwargs...)

    set_options!(opt,option_dict,kwargs...)
    if string(opt.schur_subproblem_solver) == "MadNLP.Mumps"
        @warn(logger,"When Mumps is used as a subproblem solver, Schur is run in serial.")
        @warn(logger,"To use parallelized Schur, use Ma27 or Ma57.")
    end

    inds = collect(1:nnz(csc))
    tsp = TwoStagePartition(csc,opt.schur_part,opt.schur_num_parts)

    V_0   = findall(tsp.part.==0)
    colors = get_colors(length(V_0),opt.schur_num_parts)

    csc_0,csc_0_view = get_cscsy_view(csc,V_0,inds=inds)
    schur = Matrix{Float64}(undef,length(V_0),length(V_0))

    w_0 = Vector{Float64}(undef,length(V_0))

    sws = Vector{Any}(undef,opt.schur_num_parts)

    copied_option_dict = copy(option_dict)
    @blas_safe_threads for k=1:opt.schur_num_parts
        sws[k] = SolverWorker(
            tsp,V_0,csc,inds,k,opt.schur_subproblem_solver,logger,k==1 ? option_dict : copy(copied_option_dict))
    end
    fact = opt.schur_dense_solver.Solver(schur)
    return Solver(csc,inds,tsp,schur,colors,fact,V_0,csc_0,csc_0_view,w_0,sws,opt,logger)
end

get_colors(n0,K) = [findall((x)->mod(x-1,K)+1==k,1:n0) for k=1:K]

function SolverWorker(tsp,V_0,csc::SparseMatrixCSC{Float64},inds::Vector{Int},k,
                      SubproblemSolverModule::Module,logger::Logger,option_dict::Dict{Symbol,Any})

    V    = findall(tsp.part.==k)

    csc_k,csc_k_view = get_cscsy_view(csc,V,inds=inds)
    compl,compl_view = get_csc_view(csc,V,V_0,inds=inds)
    V_0_nz = findnz(compl.colptr)

    M    = length(V) == 0 ?
        EmptyLinearSolver() : SubproblemSolverModule.Solver(csc_k;option_dict=option_dict,logger=logger)
    w    = Vector{Float64}(undef,csc_k.n)

    return SolverWorker(V,V_0_nz,csc_k,csc_k_view,compl,compl_view,M,w)
end

function findnz(colptr)
    nz = Int[]
    for j=1:length(colptr)-1
        colptr[j]==colptr[j+1] || push!(nz,j)
    end
    return nz
end

function factorize!(M::Solver)
    M.schur.=0.
    M.csc_0.nzval.=M.csc_0_view
    M.schur.=M.csc_0
    @blas_safe_threads for sw in M.sws
        sw.csc.nzval.=sw.csc_view
        sw.compl.nzval.=sw.compl_view
        factorize!(sw.M)
    end

    # asymchronous multithreading doesn't work here
    for q = 1:length(M.colors)
        @blas_safe_threads for k = 1:length(M.sws)
            for j = M.colors[mod(q+k-1,length(M.sws))+1] # each subprob works on a different color
                factorize_worker!(j,M.sws[k],M.schur)
            end
        end
    end
    factorize!(M.fact)

    return M
end

function factorize_worker!(j,sw,schur)
    j in sw.V_0_nz || return
    sw.w.= view(sw.compl,:,j)
    solve!(sw.M,sw.w)
    mul!(view(schur,:,j),sw.compl',sw.w,-1.,1.)
end


function solve!(M::Solver,x::AbstractVector{Float64})
    M.w_0 .= view(x,M.V_0)
    @blas_safe_threads for sw in M.sws
        sw.w.=view(x,sw.V)
        solve!(sw.M,sw.w)
    end
    for sw in M.sws
        mul!(M.w_0,sw.compl',sw.w,-1.,1.)
    end
    solve!(M.fact,M.w_0)
    view(x,M.V_0).=M.w_0
    @blas_safe_threads for sw in M.sws
        x_view = view(x,sw.V)
        sw.w.= x_view
        mul!(sw.w,sw.compl,M.w_0,1.,1.)
        solve!(sw.M,sw.w)
        x_view.=sw.w
    end
    return x
end

is_inertia(M::Solver) = is_inertia(M.fact) && is_inertia(M.sws[1].M)
function inertia(M::Solver)
    numpos,numzero,numneg = inertia(M.fact)
    for k=1:M.opt.schur_num_parts
        _numpos,_numzero,_numneg =  inertia(M.sws[k].M)
        numpos += _numpos
        numzero += _numzero
        numneg += _numneg
    end
    return (numpos,numzero,numneg)
end

function improve!(M::Solver)
    for sw in M.sws
        improve!(sw.M) || return false
    end
    return true
end

function introduce(M::Solver)
    for sw in M.sws
        sw.M isa EmptyLinearSolver || return "schur equipped with "*introduce(sw.M)
    end
end

end # module
