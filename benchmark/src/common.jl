function run_benchmark(model, modeloptions, cases, backend, solveroptions)
    pmap(cases) do case
        @info "Running $(string(model)) $case"
        m = model(case, backend; modeloptoins...)
        @info "$case: first run"

        result = madnlp(m; solveroptions...)
        @info "$case: first run"
        return (
            ;
            case=case,
            stats=result.counter,
        )
    end
end

