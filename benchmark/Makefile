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
	$(REV1_SHA)_cutest_gpu.jld2 \
	$(REV1_SHA)_cops_gpu.jld2 \
	$(REV1_SHA)_opf_gpu.jld2

OUT2 := $(REV2_SHA)_cutest_cpu.jld2 \
	$(REV2_SHA)_cops_cpu.jld2 \
	$(REV2_SHA)_opf_cpu.jld2 \
	$(REV2_SHA)_cutest_gpu.jld2 \
	$(REV2_SHA)_cops_gpu.jld2 \
	$(REV2_SHA)_opf_gpu.jld2

PDF  := $(REV1_SHA)_$(REV2_SHA)_cutest_gpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_cops_gpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_opf_gpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_cutest_cpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_cops_cpu.pdf \
	$(REV1_SHA)_$(REV2_SHA)_opf_cpu.pdf

.PHONY: all fetch instantiate bench1 bench2 plot clean test1

all: $(PDF)

fetch:
	@if git remote get-url origin >/dev/null 2>&1; then \
	  git fetch origin; \
	else \
	  echo "No 'origin' remote; skipping fetch"; \
	fi

instantiate:
	$(JULIA) --project -e 'import Pkg; Pkg.instantiate()'

test1:

bench1: fetch instantiate
	@echo "Running benchmark for REV1=$(REV1) [$(REV1_SHA)] with NPROCS=$(NPROCS)"
	$(JULIA) --project -p $(NPROCS) runbenchmark.jl $(REV1_SHA)
	@for f in $(OUT1); do \
	  test -f "$$f" || { echo "Missing: $$f"; exit 1; }; \
	done

bench2: fetch instantiate
	@echo "Running benchmark for REV2=$(REV2) [$(REV2_SHA)] with NPROCS=$(NPROCS)"
	$(JULIA) --project -p $(NPROCS) runbenchmark.jl $(REV2_SHA)
	@for f in $(OUT2); do \
	  test -f "$$f" || { echo "Missing: $$f"; exit 1; }; \
	done

plot: bench1 bench2
	@echo "Plotting $(REV1_SHA) vs $(REV2_SHA)"
	$(JULIA) --project plot.jl $(REV1_SHA) $(REV2_SHA)
	@test -f "$(PDF)" || (echo "Expected plot missing: $(PDF)"; exit 1)

$(PDF): plot
	@echo "Done: $(PDF)"

clean:
	rm -f *_cutest_gpu.jld2 *_cutest_gpu.pdf
