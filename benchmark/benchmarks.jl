include("benchmarks_include.jl")

@everywhere solver(nlp) = madnlp(nlp,linear_solver=MadNLPMa57,tol=1e-6)

exclude = [
    "PFIT1","PFIT2","PFIT4", # madnlp fails
    "DENSCHNE" #ipopt fails
]

probs = CUTEst.select(max_var=10)
# probs = CUTEst.select(min_var=11,max_var=50)
# probs = CUTEst.select(min_var=51,max_var=300)
# probs = CUTEst.select(min_var=301,max_var=500)
# probs = CUTEst.select(min_var=501,max_var=1000)

filter!(e->!(e in exclude),probs)

benchmark(solver,probs)

