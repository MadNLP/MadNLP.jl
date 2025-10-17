module MadNLPJuMPExt

import MadNLP, JuMP


MadNLP.@setup_workload begin
    m = JuMP.Model()
    JuMP.@variable(m, x[1:2])
    JuMP.@objective(m, Min, 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2)
    JuMP.@constraint(m, x[1] * x[2] >= 1.0)
    
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    MadNLP.@compile_workload begin
        nlp = MadNLP.HS15Model()
        MadNLP.madnlp(nlp; print_level=MadNLP.ERROR)
    end
end

end # module MadNLPJuMPExt

