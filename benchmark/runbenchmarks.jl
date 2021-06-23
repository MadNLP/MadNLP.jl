const NP = ARGS[1]

const CLASSES = filter(e-> e in ["cutest","power"], ARGS)
const SOLVERS = filter(e-> e in ["current","master","ipopt","knitro"], ARGS)

cd(@__DIR__)
cp(".Project.toml","Project.toml",force=true)

for class in CLASSES
    for solver in SOLVERS
        run(`julia --project=. benchmark-$class.jl $NP $solver`)
    end
end

run(Cmd([["julia","--project=.","plot.jl"];CLASSES;SOLVERS]))
