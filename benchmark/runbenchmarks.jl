const NP = ARGS[1]

const CLASSES = filter(e-> e in ["cutest","power"], ARGS)
const SOLVERS = filter(e-> e in ["current","master","ipopt","knitro"], ARGS)
const PROJECT_PATH = dirname(@__FILE__)
const VERBOSE = ("verbose" in ARGS) ? "verbose" : "none"

cp(
    joinpath(PROJECT_PATH, ".Project.toml"),
    joinpath(PROJECT_PATH, "Project.toml"),
    force=true
)

for class in CLASSES
    for solver in SOLVERS
        launch_script = joinpath(PROJECT_PATH, "benchmark-$class.jl")
        run(`julia --project=$PROJECT_PATH $launch_script $NP $solver $VERBOSE`)
    end
end

run(Cmd([["julia","--project=.","plot.jl"];CLASSES;SOLVERS]))
