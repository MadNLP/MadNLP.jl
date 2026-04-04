# AOT Compatibility Checker for MadNLP
#
# This script checks the MadNLP source code for patterns that are
# incompatible with ahead-of-time compilation (juliac, PackageCompiler).
#
# Usage:
#   julia aot/check_aot_compatibility.jl

const SRC_DIR = joinpath(dirname(@__DIR__), "src")
const EXT_DIR = joinpath(dirname(@__DIR__), "ext")

const BLOCKERS = String[]
const WARNINGS = String[]

function check_file(filepath::String)
    lines = readlines(filepath)
    relpath_str = relpath(filepath, dirname(@__DIR__))

    for (i, line) in enumerate(lines)
        stripped = strip(line)

        # Skip comments
        startswith(stripped, '#') && continue

        # CRITICAL: Runtime eval
        if occursin(r"\beval\(", stripped) && !occursin(r"@eval\b", stripped)
            push!(BLOCKERS, "$relpath_str:$i — runtime eval(): $stripped")
        end

        # CRITICAL: invokelatest
        if occursin(r"\binvokelatest\b|@invokelatest\b", stripped)
            push!(BLOCKERS, "$relpath_str:$i — invokelatest: $stripped")
        end

        # CRITICAL: setglobal! (dynamic global mutation)
        if occursin(r"\bsetglobal!\b", stripped)
            push!(BLOCKERS, "$relpath_str:$i — setglobal!: $stripped")
        end

        # CRITICAL: Untyped global declaration
        if occursin(r"^global\s+\w+$", stripped)
            push!(BLOCKERS, "$relpath_str:$i — untyped global: $stripped")
        end

        # WARNING: @eval (module-load-time is OK, but flag for review)
        if occursin(r"@eval\b", stripped)
            push!(WARNINGS, "$relpath_str:$i — @eval (review if load-time only): $stripped")
        end

        # WARNING: ::Type fields (type instability, not a hard blocker)
        if occursin(r"::\s*Type\b", stripped) && !occursin(r"::Type\{", stripped)
            push!(WARNINGS, "$relpath_str:$i — abstract ::Type field (type instability): $stripped")
        end

        # WARNING: Dict{Symbol, Any} (type instability)
        if occursin(r"Dict\{Symbol\s*,\s*Any\}", stripped)
            push!(WARNINGS, "$relpath_str:$i — Dict{Symbol,Any} (type instability): $stripped")
        end
    end
end

function scan_directory(dir::String)
    for (root, dirs, files) in walkdir(dir)
        for file in files
            endswith(file, ".jl") || continue
            check_file(joinpath(root, file))
        end
    end
end

println("=" ^ 60)
println("MadNLP AOT Compatibility Check")
println("=" ^ 60)
println()

scan_directory(SRC_DIR)
isdir(EXT_DIR) && scan_directory(EXT_DIR)

if isempty(BLOCKERS)
    printstyled("✓ No AOT blockers found in src/ and ext/\n\n"; color=:green)
else
    printstyled("✗ Found $(length(BLOCKERS)) AOT BLOCKER(s):\n\n"; color=:red)
    for b in BLOCKERS
        printstyled("  BLOCKER: $b\n"; color=:red)
    end
    println()
end

if isempty(WARNINGS)
    printstyled("✓ No AOT warnings\n\n"; color=:green)
else
    printstyled("⚠ Found $(length(WARNINGS)) warning(s) (not hard blockers):\n\n"; color=:yellow)
    for w in WARNINGS
        printstyled("  WARNING: $w\n"; color=:yellow)
    end
    println()
end

# Summary
println("=" ^ 60)
if isempty(BLOCKERS)
    printstyled("RESULT: MadNLP core (src/ + ext/) is AOT-compatible!\n"; color=:green, bold=true)
else
    printstyled("RESULT: MadNLP has $(length(BLOCKERS)) AOT blocker(s) to fix.\n"; color=:red, bold=true)
end
println("=" ^ 60)

exit(isempty(BLOCKERS) ? 0 : 1)
