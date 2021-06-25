using Plots

"current" in ARGS && run(`julia --project=. -p 2 benchmarks.jl current`)
"master" in ARGS && run(`julia --project=. -p 2 benchmarks.jl master`)
"ipopt" in ARGS && run(`julia --project=. -p 2 benchmarks.jl ipopt`)



