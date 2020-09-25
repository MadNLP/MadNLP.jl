hs33 = AmplModel(joinpath(@__DIR__, "hs033.nl"))
result = madnlp(hs33;print_level=MadNLP.ERROR)

@test result.status == :first_order
@test solcmp(result.solution,[0.0,1.4142135570650927,1.4142135585382265])
@test solcmp(result.multipliers,[0.17677669589725922,-0.17677669527079812])
@test solcmp(result.multipliers_L,[11.000000117266442,1.7719330023793877e-9,1.7753439380861844e-9])
@test solcmp(result.multipliers_U,[0.,0.,0.])
@test solcmp([result.objective],[-4.585786548956206])

finalize(hs33)
