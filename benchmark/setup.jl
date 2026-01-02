MADNLP_GIT_HASH = ARGS[1]
PROJECT_DIR = mktempdir()

Pkg.activate(PROJECT_DIR)
Pkg.add(name="MadNLP", rev="$MADNLP_GIT_HASH")
Pkg.add(name="MadNLPHSL", rev="$MADNLP_GIT_HASH")
Pkg.add(name="MadNLPGPU", rev="$MADNLP_GIT_HASH")

@everywhere Pkg.activate($PROJECT_DIR)
