using Pkg

MADNLP_DIR = pwd()

Pkg.update()
Pkg.resolve()
Pkg.develop([
    PackageSpec(path=MADNLP_DIR),
    PackageSpec(path=joinpath(MADNLP_DIR, "lib", "CuMadNLP")),
    PackageSpec(path=joinpath(MADNLP_DIR, "lib", "MadNLPTests")),
])
Base.compilecache(Base.identify_package("MadNLPTests"))
Base.compilecache(Base.identify_package("CuMadNLP"))
Base.compilecache(Base.identify_package("MadNLP"))
Pkg.instantiate()
