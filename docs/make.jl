using Documenter
using MadNLP

makedocs(
    sitename = "MadNLP.jl",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [MadNLP],
    repo = "https://github.com/MadNLP/MadNLP.jl/blob/{commit}{path}#{line}",
    checkdocs = :exports,
    clean=true,
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quickstart" => "quickstart.md",
        "Options" => "options.md",
        "Tutorials" => [
            "Multi-precision" => "tutorials/multiprecision.md",
            "Warm-start" => "tutorials/warmstart.md",
        ],
        "Manual" => [
            "IPM solver" => "man/solver.md",
            "KKT systems" => "man/kkt.md",
            "Linear Solvers" => "man/linear_solvers.md",
        ],
        "API Reference" => [
            "IPM solver" => "lib/ipm.md",
            "Callback wrappers" => "lib/callbacks.md",
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
