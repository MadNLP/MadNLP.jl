# MadNLP.jl.
# Created by Sungho Shin (sungho.shin@wisc.edu)

mutable struct MonolevelPartition
    g::Graph
    nparts::Int
    part::Vector{Int}
end

mutable struct MonolevelStruc
    V::Vector{Int}
    new_nbr::Vector{Int}
end

mutable struct BilevelPartition
    g_lower::Graph
    nparts_lower::Int
    part_lower::Vector{Int}
    V_lower::Vector{Vector{Int}}
    
    g_upper::Graph
    nparts_upper::Int
    part_upper::Vector{Int}
end

mutable struct BilevelStruc
    V_lower::Vector{Int}
    V_upper::Vector{Int}
    new_nbr_upper::Vector{Int}
end

mutable struct TwoStagePartition
    nparts::Int
    part::Vector{Int}
end

get_current_V(mls::MonolevelStruc) = mls.V
get_current_size(mls::MonolevelStruc) = length(mls.V)
get_full_size(mlp::MonolevelPartition) = nv(mlp.g)

function MonolevelPartition(csc::SparseMatrixCSC,part,nparts;max_size=0.)
    g   = Graph(csc)
    isempty(part) && (part=Metis.partition(g,nparts,alg=:KWAY))
    return MonolevelPartition(g,nparts,part)
end

function MonolevelStruc(mlp::MonolevelPartition,k;max_size=0.)
    V = findall(mlp.part.==k)
    new_nbr = expand!(V,mlp.g,max_size)
    return MonolevelStruc(V,new_nbr)
end

function expand!(mls::MonolevelStruc,mlp::MonolevelPartition,max_size)
    mls.new_nbr = expand!(mls.V,mlp.g,max_size,new_nbr=mls.new_nbr)
    return 
end

get_current_V(bls::BilevelStruc) = bls.V_lower
get_current_size(bls::BilevelStruc) = length(bls.V_upper)
get_full_size(blp::BilevelPartition) = blp.nparts_lower

function BilevelPartition(csc,part_lower,nparts_lower,part_upper,nparts_upper;max_size=0.)
    g_lower   = Graph(csc)
    isempty(part_lower) && (part_lower= Metis.partition(g_lower,nparts_lower,alg=:KWAY))
    V_lower = Vector{Vector{Int}}(undef,nparts_lower)
    @blas_safe_threads for k=1:nparts_lower
        V_lower[k] = findall(part_lower.==k)
    end

    g_upper = Graph(nparts_lower)
    for e in edges(g_lower)
        add_edge!(g_upper,part_lower[src(e)],part_lower[dst(e)])
    end
    isempty(part_upper) && (part_upper = Metis.partition(g_upper,nparts_upper,alg=:KWAY))
    
    return BilevelPartition(g_lower,nparts_lower,part_lower,V_lower,g_upper,nparts_upper,part_upper)
end

function BilevelStruc(blp::BilevelPartition,k;max_size=0.)
    V_upper = findall(blp.part_upper.==k)
    new_nbr_upper = expand!(V_upper,blp.g_upper,max_size)
    V_lower = vcat(blp.V_lower[V_upper]...)
    
    return BilevelStruc(V_lower,V_upper,new_nbr_upper)
end

function TwoStagePartition(csc::SparseMatrixCSC,part,nparts)
    if isempty(part) || findfirst(x->x==0.,part) == nothing
        g = Graph(csc)
        isempty(part) && (part = Metis.partition(g,nparts,alg=:KWAY))
        mark_boundary!(g,part)
    end
    return TwoStagePartition(nparts,part)
end

Graph(csc::SparseMatrixCSC) = Graph(getelistcsc(csc.colptr,csc.rowval))
getelistcsc(colptr,rowval) = [Edge(i,Int(j)) for i=1:length(colptr)-1 for j in @view rowval[colptr[i]:colptr[i+1]-1]]

function expand!(bls::BilevelStruc,blp::BilevelPartition,max_size)
    orig_size = length(bls.V_upper)
    bls.new_nbr_upper = expand!(bls.V_upper,blp.g_upper,max_size,new_nbr=bls.new_nbr_upper)
    bls.V_lower = vcat(blp.V_lower[bls.V_upper]...)
    return 
end

function expand!(V_om,g::Graph,max_size;
                 new_nbr=[])
    if isempty(new_nbr)
        new_nbr = Int[]
        for v in V_om
            append!(new_nbr,neighbors(g,v))
        end
        unique!(new_nbr)
        setdiff!(new_nbr,V_om)
    end
    
    old_nbr = V_om
    
    while (length(V_om) + length(new_nbr) < max_size) && length(V_om) < nv(g) && !isempty(new_nbr)
        append!(V_om,new_nbr)
        old_old_nbr = old_nbr
        old_nbr=new_nbr
        new_nbr = Int[]
        for v in old_nbr
            append!(new_nbr,neighbors(g,v))
        end
        unique!(new_nbr)
        setdiff!(new_nbr,old_old_nbr)
        setdiff!(new_nbr,old_nbr)
    end
    
    return new_nbr
end

function mark_boundary!(g,part)
    for e in edges(g)
        (part[src(e)]!=part[dst(e)] && part[src(e)]!= 0 && part[dst(e)] != 0) &&
            (part[src(e)] = 0; part[dst(e)] = 0)
    end
end
