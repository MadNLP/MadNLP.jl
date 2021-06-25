@everywhere using CUTEst, MadNLP, MadNLPHSL, NLPModelsIpopt

function decodemodel(name)
    finalize(CUTEstModel(name))
end

@everywhere function evalmodel(name,solver)
    println(name)
    nlp = CUTEstModel(name; decode=false)
    retval = solver(nlp)
    finalize(nlp)
    retval
end

function benchmark(solver,probs)
    println("Decoding problems")
    decodemodel.(probs)
    println("Solving problems")
    retvals = pmap(prob->evalmodel(prob,solver),probs)
    time   = [retval.elapsed_time for retval in retvals]
    status = [retval.status       for retval in retvals]
    time,status
end
