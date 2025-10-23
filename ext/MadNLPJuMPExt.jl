module MadNLPJuMPExt

using MadNLP, MathOptInterface, JuMP, NLPModels, PrecompileTools

PrecompileTools.@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.

    PrecompileTools.@compile_workload begin
        nlp = MadNLP.HS15Model()
        MadNLP.madnlp(nlp; print_level=MadNLP.ERROR)
        precompile(Tuple{typeof(MadNLP.madnlp), NLPModels.AbstractNLPModel{T, S} where S where T})
    end
end

end # module MadNLPJuMPExt

