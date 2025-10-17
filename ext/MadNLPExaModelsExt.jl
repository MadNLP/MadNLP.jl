module MadNLPExaModelsExt

import MadNLP, ExaModels

MadNLP.@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    MadNLP.@compile_workload begin
        nlp = MadNLP.HS15Model()
        MadNLP.madnlp(nlp; print_level=MadNLP.ERROR)
    end
end

end # module MadNLPExaModelsExt
