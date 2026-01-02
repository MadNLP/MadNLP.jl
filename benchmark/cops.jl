using COPSBenchmark

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

