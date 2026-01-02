using Plots, BenchmarkProfiles, JLD2

REV1 = ARGS[1]
REV2 = ARGS[2]

@info "Loading results for revision $rev"

for class in ["opf", "cops", "cutest"]
    for backend in ["cpu", "gpu"]
        
        @load "$(REV1)_$(class)_$(backend).jld2" results1
        @load "$(REV2)_$(class)_$(backend).jld2" results2

        for metric in [:total_time]
            plt = performance_profile(
                PlotsBackend(),
                hcat(
                    [getproperty(r.stats, metric) for r in results1],
                    [getproperty(r.stats, metric) for r in results2]
                ),
                ["$REV1", "$REV2"],
                title="$(metric), $(class) on $(backend)",
            )
            savefig(plt, "$(metric)_$(class)_$(backend).png")
        end
    end
end
