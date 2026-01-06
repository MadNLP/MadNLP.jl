# Defaults
NPROCS ?= 24
JULIA  ?= julia

# REV1/REV2 defaults are refs; we resolve them to short hashes below.
REV1 ?= HEAD
REV2 ?= origin/master

# Resolve refs -> 7-char hashes (keeps things consistent in filenames)
REV1_SHA := $(shell git rev-parse --short=7 $(REV1))
REV2_SHA := $(shell git rev-parse --short=7 $(REV2))

OUT1 := $(REV1_SHA)_cutest_cpu.jld2 \
	$(REV1_SHA)_cops_cpu.jld2 \
	$(REV1_SHA)_opf_cpu.jld2 \
	$(REV1_SHA)_cops_gpu.jld2 \
	$(REV1_SHA)_opf_gpu.jld2

OUT2 := $(REV2_SHA)_cutest_cpu.jld2 \
	$(REV2_SHA)_cops_cpu.jld2 \
	$(REV2_SHA)_opf_cpu.jld2 \
	$(REV2_SHA)_cops_gpu.jld2 \
	$(REV2_SHA)_opf_gpu.jld2

PDF  := $(REV1_SHA)_$(REV2_SHA)_total_time_cops_gpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_total_time_opf_gpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_total_time_cutest_cpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_total_time_cops_cpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_total_time_opf_cpu.pdf

.PHONY: all fetch instantiate bench1 bench2 plot clean test1

all: $(PDF)

fetch:
	@if git remote get-url origin >/dev/null 2>&1; then \
	  git fetch origin; \
	else \
	  echo "No 'origin' remote; skipping fetch"; \
	fi

bench1: fetch
	@echo "Instantiating packages for REV1=$(REV1) [$(REV1_SHA)]"
	$(JULIA) --project -e 'import Pkg; Pkg.instantiate(); Pkg.add(name="MadNLP", rev="$(REV1_SHA)"); Pkg.add(name="MadNLPHSL", rev="$(REV1_SHA)"); Pkg.add(name="MadNLPGPU", rev="$(REV1_SHA)")'
	@echo "Running benchmark for REV1=$(REV1) [$(REV1_SHA)] with NPROCS=$(NPROCS)"
	$(JULIA) --project -p $(NPROCS) runbenchmark.jl $(REV1_SHA)
	@for f in $(OUT1); do \
	  test -f "$$f" || { echo "Missing: $$f"; exit 1; }; \
	done

bench2: fetch
	@echo "Instantiating packages for REV2=$(REV2) [$(REV2_SHA)]"
	$(JULIA) --project -e 'import Pkg; Pkg.instantiate(); Pkg.add(name="MadNLP", rev="$(REV2_SHA)"); Pkg.add(name="MadNLPHSL", rev="$(REV2_SHA)"); Pkg.add(name="MadNLPGPU", rev="$(REV2_SHA)")'
	@echo "Running benchmark for REV2=$(REV2) [$(REV2_SHA)] with NPROCS=$(NPROCS)"
	$(JULIA) --project -p $(NPROCS) runbenchmark.jl $(REV2_SHA)
	@for f in $(OUT2); do \
	  test -f "$$f" || { echo "Missing: $$f"; exit 1; }; \
	done

# plot: bench1 bench2
plot:
	@echo "Generating plots for REV1=$(REV1_SHA) and REV2=$(REV2_SHA)"
	$(JULIA) --project plot.jl $(REV1_SHA) $(REV2_SHA)
	@for f in $(PDF); do \
	  test -f "$$f" || { echo "Missing plot: $$f"; exit 1; }; \
	done

$(PDF): plot
	@echo "Done: $(PDF)"

clean: 
	rm -f *.jld2 *.pdf
