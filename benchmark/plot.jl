using Plots, BenchmarkProfiles, JLD2

REV1 = ARGS[1]
REV2 = ARGS[2]

for (class, backend) in [
    ("opf","cpu")
    ("cops","cpu")
    ("cutest","cpu")
    ("opf","gpu")
    ("cops","gpu")
    ]
    
    results1 = JLD2.load("$(REV1)_$(class)_$(backend).jld2")["results"]
    results2 = JLD2.load("$(REV2)_$(class)_$(backend).jld2")["results"]

    for metric in [:total_time]
        plt = performance_profile(
            PlotsBackend(),
            hcat(
                [getproperty(r, metric) for r in results1],
                [getproperty(r, metric) for r in results2]
            ),
            ["$REV1", "$REV2"],
            title="$(metric), $(class) on $(backend)",
        )
        savefig(plt, "$(REV1)_$(REV2)_$(metric)_$(class)_$(backend).pdf")
    end
end


