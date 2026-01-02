using CUTEst

const CUTEST_CASES = setdiff(
    CUTEst.select_sif_problems()[1:50],
    [
        # MadNLP running into error
        # Ipopt running into error
        "EG3", # lfact blows up
        # Problems that are hopelessly large
        "TAX213322",
        "TAXR213322",
        "TAX53322",
        "TAXR53322",
        "YATP1LS",
        "YATP2LS",
        "YATP1CLS",
        "YATP2CLS",
        "CYCLOOCT",
        "CYCLOOCF",
        "LIPPERT1",
        "GAUSSELM",
        "BA-L52LS",
        "BA-L73LS",
        "BA-L21LS",
        "BA-L52",
        "NONSCOMPNE",
        "LOBSTERZ",
        # Failure
        "CHARDIS0"
    ]
)

try
    @info "Testing CUTEst model loading..."
    m = CUTEstModel(CUTEst.select_sif_problems()[end]; decode = false)
catch e
    @info "CUTEst models could not be loaded. Decoding all CUTEst problems..."
    for (i, instance) in enumerate(CUTEst.select_sif_problems())
        try
            m=CUTEstModel(instance; decode = false)
            @debug "Model $i-th $(instance) loaded successfully."
            finalize(m)
        catch e
            CUTEst.sifdecoder(instance)
            CUTEst.build_libsif(instance)
            m=CUTEstModel(instance; decode = false)
            finalize(m)
            @info "Model $i-th $(instance) decoded and loaded successfully."
        end
    end
end
