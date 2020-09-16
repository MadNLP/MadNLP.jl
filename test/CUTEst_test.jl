using CUTEst

selected_case = ["DIXMAANI","LIARWHD","SCHMVETT","POLAK4","DUAL2","MPC2","BLOWEYA","HS35",
                 "CMPC3","ACOPP300","AUG2D","LISWET9","ZECEVIC3","GMNCASE2","LUKVLE8",
                 "READING2","MAXLIKA","HYDROELM","DIXMAANJ","AVGASB","PALMER8A","OBSTCLAE",
                 "READING6","HS7","SPIN2LS","SBRYBND","ARGLINC","TOINTGOR","S268","DIXMAANC",
                 "TABLE8","BROWNDEN","HILBERTA","STEENBRA","BQPGAUSS","DUALC5"]
kwargs_collection = [
    Dict(:inertia_correction_method=>"inertia_free"),
    Dict(:inertia_correction_method=>"inertia_based")]

GC.gc()
@testset "CUTEst $str" for str in selected_case
    for kwargs in kwargs_collection
        @isdefined(model) && cutest_finalize(model)
        model = CUTEstModel(str)
        result= madnlp(model;kwargs...)
        @test result.status in [:first_order,:acceptable,:infeasible]
    end
end

