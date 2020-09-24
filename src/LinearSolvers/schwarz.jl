# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module Schwarz

import ..MadNLP:
    @with_kw, Logger, @debug, @warn, @error,
    default_linear_solver,SparseMatrixCSC, SubVector, StrideOneVector, get_cscsy_view, nnz,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    AbstractOptions, AbstractLinearSolver, set_options!,
    MonolevelPartition, MonolevelStruc, BilevelPartition, BilevelStruc,
    expand!, get_current_V, get_current_size, get_full_size,
    EmptyLinearSolver, introduce, factorize!, solve!, improve!, is_inertia, inertia,
    set_blas_num_threads, blas_num_threads, @blas_safe_threads, @sprintf

const INPUT_MATRIX_TYPE = :csc

@with_kw mutable struct Options <: AbstractOptions
    schwarz_num_parts_upper::Int = 0
    schwarz_num_parts::Int = 2
    schwarz_subproblem_solver::Module
    schwarz_fully_improve_subproblem_solver::Bool=true
    schwarz_max_expand_factor::Int = 4
    schwarz_custom_partition::Bool = false
    schwarz_part_upper::Vector{Int} = Int[]
    schwarz_part::Vector{Int} = Int[]
end

# mutable struct EmptySolverWorker end
# factorizeEmptySolverWorker

mutable struct SolverWorker
    struc::Union{MonolevelStruc,BilevelStruc}
    
    max_size::Float64
    overlap_increase_factor::Float64
    restrictor::Vector{Int}
    
    csc::SparseMatrixCSC{Float64,Int32}
    csc_view::SubVector{Float64}
    
    M::AbstractLinearSolver
    
    p::AbstractVector{Float64}
    q::AbstractVector{Float64}
    q_restricted::AbstractVector{Float64}
end

mutable struct Solver <: AbstractLinearSolver
    partition::Union{MonolevelPartition,BilevelPartition}
    
    csc::SparseMatrixCSC{Float64}
    inds::Vector{Int}
    
    p::Vector{Float64}
    w::Vector{Float64}
    sws::Vector{SolverWorker}
    opt::Options
    logger::Logger
end


function Solver(csc::SparseMatrixCSC{Float64};
                option_dict::Dict{Symbol,Any}=Dict(),
                opt=Options(schwarz_subproblem_solver=default_linear_solver()),logger=Logger(),
                kwargs...)
    
    set_options!(opt,option_dict,kwargs...)

    inds = collect(1:nnz(csc))
    
    p = zeros(csc.n)
    w = zeros(csc.n)
    
    if opt.schwarz_num_parts_upper == 0
        partition = MonolevelPartition(
            csc,opt.schwarz_custom_partition ? opt.schwarz_part : Int32[],opt.schwarz_num_parts) 
    else
        partition = BilevelPartition(
            csc,
            opt.schwarz_custom_partition ? opt.schwarz_part : Int32[],opt.schwarz_num_parts,
            opt.schwarz_custom_partition ? opt.schwarz_part_upper : Int32[],opt.schwarz_num_parts_upper)
    end

    
    sws=Vector{SolverWorker}(undef,opt.schwarz_num_parts_upper == 0 ? opt.schwarz_num_parts : opt.schwarz_num_parts_upper)
    copied_option_dict = copy(option_dict)
    @blas_safe_threads for k=1:length(sws)
        sws[k] = SolverWorker(
            partition,csc,inds,p,k,opt.schwarz_max_expand_factor,
            opt.schwarz_subproblem_solver,opt.schwarz_fully_improve_subproblem_solver,
            logger,k==1 ? option_dict : copy(copied_option_dict))
    end

    saturation = maximum(sws[k].csc.n/csc.n*100 for k=1:length(sws))
    @debug(logger,@sprintf("overlap size initialized with %3d%% saturation.\n",saturation))
    
    return Solver(partition,csc,inds,p,w,sws,opt,logger)
end

function SolverWorker(
    partition,csc::SparseMatrixCSC{Float64},inds::Vector{Int},p::Vector{Float64},
    k::Int,max_expand_factor::Int,SubproblemSolverModule::Module,fully_improve_subproblem_solver::Bool,
    logger::Logger,option_dict::Dict{Symbol,Any})
    struc= partition isa MonolevelPartition ?
        MonolevelStruc(partition,k) : BilevelStruc(partition,k)
    overlap_increase_factor =
        (get_full_size(partition)/get_current_size(struc))^(1/max_expand_factor)
    max_size = get_current_size(struc)*(overlap_increase_factor)    
    p = view(p,copy(get_current_V(struc)))
    restrictor = 1:length(get_current_V(struc))
    expand!(struc,partition,max_size)
    csc,csc_view = get_cscsy_view(csc,get_current_V(struc),inds=inds)
    if csc.n == 0
        M = EmptyLinearSolver()
        @warn(logger,"empty subproblem at partition $k")
    else
        M = SubproblemSolverModule.Solver(csc;option_dict = option_dict,logger=logger)
    end
    fully_improve_subproblem_solver && while improve!(M) end # starts with fully improved    
    q = Vector{Float64}(undef,length(get_current_V(struc)))
    q_restricted = view(q,restrictor)
    
    return SolverWorker(struc,max_size,overlap_increase_factor,restrictor,csc,csc_view,M,p,q,q_restricted)
end

function factorize!(M::Solver)
    @blas_safe_threads for k=1:length(M.sws)
        M.sws[k].csc.nzval.=M.sws[k].csc_view
        factorize!(M.sws[k].M)
    end
    return M
end

is_maximal_overlap(M::Solver) = all(sw.max_size>=M.csc.n for sw in M.sws)


function improve!(M::Solver)
    is_maximal_overlap(M) && (@debug(M.logger,"improve quality failed.\n");return false)
    
    @blas_safe_threads for k=1:length(M.sws)
        M.sws[k].max_size *= M.sws[k].overlap_increase_factor
        expand!(M.sws[k].struc,M.partition,M.sws[k].max_size)
        M.sws[k].csc,M.sws[k].csc_view = get_cscsy_view(M.csc,get_current_V(M.sws[k].struc);inds=M.inds)
        if M.sws[k].csc.n == 0
            M.sws[k].M = EmptyLinearSolver()
        else
            M.sws[k].M = M.opt.schwarz_subproblem_solver.Solver(M.sws[k].csc;opt=M.sws[k].M.opt)
        end
        M.opt.schwarz_fully_improve_subproblem_solver && while improve!(M.sws[k].M) end
        resize!(M.sws[k].q,length(get_current_V(M.sws[k].struc)))
    end
    
    saturation = maximum(M.sws[k].csc.n/M.csc.n*100 for k=1:length(M.sws));
    saturation == 100. ?
        @warn(M.logger,@sprintf("overlap is maximally saturated")) :
        @debug(M.logger,@sprintf("overlap size increased to %3d%% saturation.\n",saturation)) 
        
    
    return true
end

is_inertia(::Solver)=false
function inertia(M::Solver)
    throw(InertiaException)
end

function solve!(M::Solver,x::AbstractVector{Float64})
    @blas_safe_threads for k=1:length(M.sws)
        M.sws[k].q.=view(x,get_current_V(M.sws[k].struc))
        solve!(M.sws[k].M,M.sws[k].q)
        M.sws[k].p.=M.sws[k].q_restricted
    end
    x.=M.p
end

introduce(M::Solver)="schwarz equipped with "*introduce(M.sws[1].M)

end # module

# forgiving names
const schwarz=Schwarz;
const SCHWARZ=Schwarz;

