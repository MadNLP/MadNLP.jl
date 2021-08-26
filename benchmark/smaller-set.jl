using CUTEst, DelimitedFiles

probs= CUTEst.select()


similar(a,b;atol=10,rtol=0.5) = (abs(a-b) < atol) || (abs(a-b)/max(a,b) < rtol)

function is_similar_exists(bin,nlp)
    for (name,nvar,ncon,nnzh,nnzj) in bin
        if similar(nvar,nlp.meta.nvar) && similar(ncon,nlp.meta.ncon) && similar(nnzh,nlp.meta.nnzh) && similar(nnzj,nlp.meta.nnzj)
            return true
        end
    end
    return false
end

bin = Tuple{String,Int,Int,Int,Int}[]


for i=1:length(probs)
    prob = probs[i]
    nlp = CUTEstModel(prob)
    finalize(nlp)
    
    if is_similar_exists(bin,nlp)
        println("#$cnt Skipping $(nlp.meta.name)")
    else
        println("#$cnt Putting $(nlp.meta.name) in the bin")
        push!(bin, (nlp.meta.name,nlp.meta.nvar,nlp.meta.ncon,nlp.meta.nnzh,nlp.meta.nnzj))
    end
end

writedlm("cutest-quick-names.csv",[name for (name,nvar,ncon,nnzh,nnzj) in bin],',')
