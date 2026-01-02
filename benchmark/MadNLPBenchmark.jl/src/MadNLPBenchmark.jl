module MadNLPBenchmark

using Distributed, JLD2, Logging
using ExaModelsPower, COPSBenchmark, CUTEst

const OPF_CASES = [
    "pglib_opf_case197_snem.m", " pglib_opf_case300_ieee.m", " pglib_opf_case588_sdet.m",
    "pglib_opf_case2000_goc.m", " pglib_opf_case3012wp_k.m", " pglib_opf_case5_pjm.m",
    "pglib_opf_case200_activ.m", " pglib_opf_case3022_goc.m", " pglib_opf_case60_c.m",
    "pglib_opf_case20758_epigrids.m", " pglib_opf_case30_as.m", " pglib_opf_case6468_rte.m",
    "pglib_opf_case2312_goc.m", " pglib_opf_case30_ieee.m", " pglib_opf_case6470_rte.m",
    "pglib_opf_case2383wp_k.m", " pglib_opf_case3120sp_k.m", " pglib_opf_case6495_rte.m",
    "pglib_opf_case10000_goc.m", " pglib_opf_case240_pserc.m", " pglib_opf_case3375wp_k.m", " pglib_opf_case6515_rte.m",
    "pglib_opf_case10192_epigrids.m", " pglib_opf_case24464_goc.m", " pglib_opf_case3970_goc.m", " pglib_opf_case7336_epigrids.m",
    "pglib_opf_case10480_goc.m", " pglib_opf_case24_ieee_rts.m", " pglib_opf_case39_epri.m", " pglib_opf_case73_ieee_rts.m",
    "pglib_opf_case118_ieee.m", " pglib_opf_case2736sp_k.m", " pglib_opf_case3_lmbd.m", " pglib_opf_case78484_epigrids.m",
    "pglib_opf_case1354_pegase.m", " pglib_opf_case2737sop_k.m", " pglib_opf_case4020_goc.m", " pglib_opf_case793_goc.m",
    "pglib_opf_case13659_pegase.m", " pglib_opf_case2742_goc.m", " pglib_opf_case4601_goc.m", " pglib_opf_case8387_pegase.m",
    "pglib_opf_case14_ieee.m", " pglib_opf_case2746wop_k.m", " pglib_opf_case4619_goc.m", " pglib_opf_case89_pegase.m",
    "pglib_opf_case162_ieee_dtc.m", " pglib_opf_case2746wp_k.m", " pglib_opf_case4661_sdet.m", " pglib_opf_case9241_pegase.m",
    "pglib_opf_case179_goc.m", " pglib_opf_case2848_rte.m", " pglib_opf_case4837_goc.m", " pglib_opf_case9591_goc.m",
    "pglib_opf_case1803_snem.m", " pglib_opf_case2853_sdet.m", " pglib_opf_case4917_goc.m", " README.md",
    "pglib_opf_case1888_rte.m", " pglib_opf_case2868_rte.m", " pglib_opf_case500_goc.m", " sad",
    "pglib_opf_case19402_goc.m", " pglib_opf_case2869_pegase.m", " pglib_opf_case5658_epigrids.m",
    "pglib_opf_case1951_rte.m", " pglib_opf_case30000_goc.m", " pglib_opf_case57_ieee.m"
]

const COPS_CASES = [
    (COPSBenchmark.bearing_model, (50, 50)),
    (COPSBenchmark.chain_model, (800,)),
    (COPSBenchmark.camshape_model, (1000,)),
    (COPSBenchmark.catmix_model, (100,)),
    (COPSBenchmark.elec_model, (50,)),
    (COPSBenchmark.gasoil_model, (100,)),
    (COPSBenchmark.marine_model, (100,)),
    (COPSBenchmark.methanol_model, (100,)),
    (COPSBenchmark.minsurf_model, (50, 50)),
    (COPSBenchmark.minsurf_model, (50, 75)),
    (COPSBenchmark.minsurf_model, (50, 100)),
    (COPSBenchmark.pinene_model, (100,)),
    (COPSBenchmark.robot_model, (200,)),
    (COPSBenchmark.rocket_model, (400,)),
    (COPSBenchmark.steering_model, (200,)),
    (COPSBenchmark.dirichlet_model, (20,)),
    (COPSBenchmark.henon_model, (10,)),
    (COPSBenchmark.lane_emden_model, (20,)),
]

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

function __init__()
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
end

function run_benchmark(name, model, cases, solver, analyzer; wks=workers(), finalizer=m -> nothing)
    results = []
    pmap(WorkerPool(wks), cases) do case
        @info "Running $(name) $case"
        m = model(case)
        @info "$case: first run"
        solver(m)
        @info "$case: second run"
        result = solver(m)
        @info "$case: completed"
        finalizer(m)
    end
    JLD2.@save "$name.jld2" name results
end

export run_benchmark, OPF_CASES, COPS_CASES, CUTEST_CASES

end # module
