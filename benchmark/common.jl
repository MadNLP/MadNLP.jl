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
