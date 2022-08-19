using Documenter
using Pkg
using MadNLP

Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "../lib/MadNLPTests/")))
Pkg.instantiate()

makedocs(
    sitename = "MadNLP.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [MadNLP],
    repo = "https://github.com/MadNLP/MadNLP.jl/blob/{commit}{path}#{line}",
    strict = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quickstart" => "quickstart.md",
        "Options" => "options.md",
        "Manual" => [
            "Solver" => "man/solver.md",
            "KKT systems" => "man/kkt.md",
            "Linear Solvers" => "man/linear_solvers.md",
        ],
        "API Reference" => [
            "KKT systems" => "lib/kkt.md",
            "Linear Solvers" => "lib/linear_solvers.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/MadNLP/MadNLP.jl.git",
    target = "build",
    devbranch = "master",
    devurl = "dev",
    push_preview = true,
)
