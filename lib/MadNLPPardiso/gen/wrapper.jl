# Script to parse the Pardiso header and generate Julia wrappers.
using Clang
using Clang.Generators
using JuliaFormatter

function main()
  @info "Wrapping Pardiso"

  cd(@__DIR__)
  include_dir = joinpath(pwd(), "include")
  headers = joinpath(include_dir, "pardiso.h")

  options = load_options(joinpath(@__DIR__, "generator.toml"))
  options["general"]["output_file_path"] = joinpath("..", "src", "libpardiso.jl")
  options["general"]["library_name"] = "libpardiso"
  options["general"]["output_ignorelist"] = ["doublecomplex"]

  args = get_default_args()
  push!(args, "-I$include_dir")

  ctx = create_context(headers, args, options)
  build!(ctx)

  path = options["general"]["output_file_path"]
  format_file(path, YASStyle())

  text = read(path, String)
  # text = replace(text, "Clong" => "SS_Int")
  # text = replace(text, "end\n\nconst" => "end\n\n\nconst")
  # text = replace(text, "\n\nconst" => "\nconst")
  write(path, text)
  return nothing
end

# If we want to use the file as a script with `julia wrapper.jl`
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
