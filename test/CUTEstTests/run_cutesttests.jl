import MadNLP, CUTEst
using Test
    
selected_case = ["DIXMAANI","LIARWHD","SCHMVETT","POLAK4","DUAL2","MPC2","HS35",
                 "CMPC3","ACOPP300","AUG2D","LISWET9","ZECEVIC3","GMNCASE2","LUKVLE8",
                 "READING2","MAXLIKA","HYDROELM","DIXMAANJ","AVGASB","PALMER8A","OBSTCLAE",
                 "READING6","HS7","SPIN2LS","SBRYBND","ARGLINC","TOINTGOR","S268","DIXMAANC",
                 "TABLE8","BROWNDEN","HILBERTA","STEENBRA","BQPGAUSS","DUALC5"]
kwargs_collection = [
    Dict(),
    Dict(:iterator=>MadNLP.Krylov,:print_level=>MadNLP.ERROR),
    Dict(:reduced_system=>false,:print_level=>MadNLP.ERROR),
    Dict(:fixed_variable_treatment=>MadNLP.RELAX_BOUND,:print_level=>MadNLP.ERROR),
    # Dict(:inertia_correction_method=>MadNLP.INERTIA_FREE,:print_level=>MadNLP.ERROR),
    # Dict(:inertia_correction_method=>MadNLP.INERTIA_BASED,:print_level=>MadNLP.ERROR)
]

@testset "CUTEst" begin
    for str in selected_case
        for kwargs in kwargs_collection
            model = CUTEst.CUTEstModel(str)
            result = nothing
            try
                result= MadNLP.madnlp(model;kwargs...)
            finally
                finalize(model)
            end
            @test @isdefined(result) && result.status in [:first_order,:acceptable,:infeasible]
        end
    end
end

