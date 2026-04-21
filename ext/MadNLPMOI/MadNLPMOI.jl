module MadNLPMOI

import MadNLP
import NLPModels
import MathOptInterface as MOI
import MathOptInterface.Utilities as MOIU

function __init__()
    MadNLP._optimizer_type[] = Optimizer
    return
end

include("MOI_utils.jl")
include("MOI_wrapper.jl")

end # module MadNLPMOI
