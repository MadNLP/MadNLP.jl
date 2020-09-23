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

@enum(FixedVariableTreatment::Int,
      RELAX_BOUND = 1,
      MAKE_PARAMETER = 2)

@enum(InertiaCorrectionMethod::Int,
      INERTIA_AUTO = 1,
      INERTIA_BASED = 2,
      INERTIA_FREE = 3)

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
      INTERNAL_ERROR = -6)

const status_output_dict = Dict(
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
    INTERNAL_ERROR => "Internal Error.")

const status_moi_dict = Dict(
    SOLVE_SUCCEEDED => MOI.LOCALLY_SOLVED,
    SOLVED_TO_ACCEPTABLE_LEVEL => MOI.ALMOST_LOCALLY_SOLVED,
    SEARCH_DIRECTION_BECOMES_TOO_SMALL => MOI.SLOW_PROGRESS,
    DIVERGING_ITERATES => MOI.INFEASIBLE_OR_UNBOUNDED,
    INFEASIBLE_PROBLEM_DETECTED => MOI.LOCALLY_INFEASIBLE,
    MAXIMUM_ITERATIONS_EXCEEDED => MOI.ITERATION_LIMIT,
    MAXIMUM_WALLTIME_EXCEEDED => MOI.TIME_LIMIT,
    INITIAL => MOI.OPTIMIZE_NOT_CALLED,
    RESTORATION_FAILED => MOI.NUMERICAL_ERROR,
    INVALID_NUMBER_DETECTED => MOI.INVALID_MODEL,
    ERROR_IN_STEP_COMPUTATION => MOI.NUMERICAL_ERROR,
    NOT_ENOUGH_DEGREES_OF_FREEDOM => MOI.INVALID_MODEL,
    USER_REQUESTED_STOP => MOI.INTERRUPTED,
    INTERNAL_ERROR => MOI.OTHER_ERROR)
