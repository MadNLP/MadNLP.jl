# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

# Options
@enum(LogLevels::Int,
      TRACE  = 1,
      DEBUG  = 2,
      INFO   = 3,
      NOTICE = 4,
      WARN   = 5,
      ERROR  = 6)

@enum(FixedVariableTreatments::Int,
      RELAX_BOUND = 1,
      MAKE_PARAMETER = 2)

@enum(InertiaCorrectionMethod::Int,
      INERTIA_AUTO = 1,
      INERTIA_BASED = 2,
      INERTIA_FREE = 3)

@enum(KKTLinearSystem::Int,
      SPARSE_KKT_SYSTEM = 1,
      SPARSE_UNREDUCED_KKT_SYSTEM = 2,
      DENSE_KKT_SYSTEM = 3,
      DENSE_CONDENSED_KKT_SYSTEM = 4,
)

@enum(HessianApproximation::Int,
      EXACT_HESSIAN = 1,
      DENSE_BFGS = 2,
      DENSE_DAMPED_BFGS = 3,
      SPARSE_COMPACT_LBFGS = 4,
)

@enum(BFGSInitStrategy::Int,
      SCALAR1  = 1,
      SCALAR2  = 2,
      SCALAR3  = 3,
      SCALAR4  = 4,
      CONSTANT = 5,
)

@enum(Status::Int,
      SOLVE_SUCCEEDED = 1,
      SOLVED_TO_ACCEPTABLE_LEVEL = 2,
      SEARCH_DIRECTION_BECOMES_TOO_SMALL = 3,
      DIVERGING_ITERATES = 4,
      INFEASIBLE_PROBLEM_DETECTED = 5,
      MAXIMUM_ITERATIONS_EXCEEDED = 6,
      MAXIMUM_WALLTIME_EXCEEDED = 7,
      INITIAL = 11,
      REGULAR = 12,
      RESTORE = 13,
      ROBUST  = 14,
      RESTORATION_FAILED = -1,
      INVALID_NUMBER_DETECTED = -2,
      ERROR_IN_STEP_COMPUTATION = -3,
      NOT_ENOUGH_DEGREES_OF_FREEDOM = -4,
      USER_REQUESTED_STOP = -5,
      INTERNAL_ERROR = -6,
      INVALID_NUMBER_OBJECTIVE = -7,
      INVALID_NUMBER_GRADIENT = -8,
      INVALID_NUMBER_CONSTRAINTS = -9,
      INVALID_NUMBER_JACOBIAN = -10,
      INVALID_NUMBER_HESSIAN_LAGRANGIAN = -11,
)

const STATUS_OUTPUT_DICT = Dict(
    SOLVE_SUCCEEDED => "Optimal Solution Found.",
    SOLVED_TO_ACCEPTABLE_LEVEL => "Solved To Acceptable Level.",
    SEARCH_DIRECTION_BECOMES_TOO_SMALL => "Search Direction is becoming Too Small.",
    DIVERGING_ITERATES => "Iterates divering; problem might be unbounded.",
    MAXIMUM_ITERATIONS_EXCEEDED => "Maximum Number of Iterations Exceeded.",
    MAXIMUM_WALLTIME_EXCEEDED => "Maximum wall-clock Time Exceeded.",
    RESTORATION_FAILED => "Restoration Failed",
    INFEASIBLE_PROBLEM_DETECTED => "Converged to a point of local infeasibility. Problem may be infeasible.",
    INVALID_NUMBER_DETECTED => "Invalid number in NLP function or derivative detected.",
    ERROR_IN_STEP_COMPUTATION => "Error in step computation.",
    NOT_ENOUGH_DEGREES_OF_FREEDOM => "Problem has too few degrees of freedom.",
    USER_REQUESTED_STOP => "Stopping optimization at current point as requested by user.",
    INTERNAL_ERROR => "Internal Error.",
    INVALID_NUMBER_OBJECTIVE => "Invalid number in NLP objective function detected.",
    INVALID_NUMBER_GRADIENT => "Invalid number in NLP objective gradient detected.",
    INVALID_NUMBER_CONSTRAINTS => "Invalid number in NLP constraint function detected.",
    INVALID_NUMBER_JACOBIAN => "Invalid number in NLP constraint Jacobian detected.",
    INVALID_NUMBER_HESSIAN_LAGRANGIAN => "Invalid number in NLP Hessian Lagrangian detected.",
)
