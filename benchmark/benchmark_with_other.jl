solvers = Dict(
    "madnlp"=> nlp -> madnlp(nlp,linear_solver=MadNLPMa57,tol=1e-6),
    "ipopt" => nlp -> ipopt(nlp,linear_solver="ma57",tol=1e-6)
)

