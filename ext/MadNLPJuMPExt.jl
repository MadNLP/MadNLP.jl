module MadNLPJuMPExt

using MadNLP, MathOptInterface, JuMP, NLPModels

MadNLP.@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.

    MadNLP.@compile_workload begin
        m = Model(MadNLP.Optimizer)
        @variable(m, x)
        @constraint(m, exp(x) >= 0)
        @objective(m, Min, (x-1)^4)
        optimize!(m)
        
        nlp = MadNLP.HS15Model()
        MadNLP.madnlp(nlp; print_level=MadNLP.ERROR)
        precompile(Tuple{typeof(MadNLP.madnlp), NLPModels.AbstractNLPModel{T, S} where S where T})
    end
end

end # module MadNLPJuMPExt

