using Pkg

MADNLP_DIR = pwd()

Pkg.update()
Pkg.resolve()
Pkg.add("https://github.com/JuliaBinaryWrappers/MbedTLS_jll.jl.git", rev="main")
Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPTests"))
Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPGPU"))
Pkg.develop(path=MADNLP_DIR)
Base.compilecache(Base.identify_package("MadNLPTests"))
Base.compilecache(Base.identify_package("MadNLPGPU"))
Base.compilecache(Base.identify_package("MadNLP"))
Pkg.instantiate()
