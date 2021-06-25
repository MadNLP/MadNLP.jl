using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.DeprecatedTest
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

const config = MOIT.Config(atol=1e-4, rtol=1e-4,
                           optimal_status=MOI.LOCALLY_SOLVED)
const config_no_duals = MOIT.Config(atol=1e-4, rtol=1e-4, duals=false,
                                    optimal_status=MOI.LOCALLY_SOLVED)

@testset "MOI utils" begin
    optimizer = MadNLP.Optimizer()
    @testset "SolverName" begin
        @test MOI.get(optimizer, MOI.SolverName()) == "MadNLP"
    end
    @testset "supports_default_copy_to" begin
        @test MOI.supports_incremental_interface(optimizer)
    end
    @testset "MOI.Silent" begin
        @test MOI.supports(optimizer, MOI.Silent())
        @test MOI.get(optimizer, MOI.Silent()) == false
        MOI.set(optimizer, MOI.Silent(), true)
        @test MOI.get(optimizer, MOI.Silent()) == true
    end
    @testset "MOI.TimeLimitSec" begin
        @test MOI.supports(optimizer, MOI.TimeLimitSec())
        my_time_limit = 10.
        MOI.set(optimizer, MOI.TimeLimitSec(), my_time_limit)
        @test MOI.get(optimizer, MOI.TimeLimitSec()) == my_time_limit
    end
    @testset "MOI.MaxIter" begin
        MOI.set(optimizer,MOI.RawOptimizerAttribute("max_iter"),1)
        @test MOI.get(optimizer,MOI.RawOptimizerAttribute("max_iter")) == 1
    end
end

# Currently broken on MOI 0.10
# See: https://github.com/jump-dev/MathOptInterface.jl/pull/1591
# @testset "Testing getters" begin
#     MOIT.copytest(MOI.instantiate(()->MadNLP.Optimizer(print_level=MadNLP.ERROR),
#                                   with_bridge_type=Float64), MOIU.Model{Float64}())
# end

@testset "Bounds set twice" begin
    optimizer = MadNLP.Optimizer(print_level=MadNLP.ERROR)
    MOIT.set_lower_bound_twice(optimizer, Float64)
    MOIT.set_upper_bound_twice(optimizer, Float64)
end

@testset "MOI Linear tests" begin
    optimizer = MadNLP.Optimizer(print_level=MadNLP.ERROR)
    exclude = ["linear1", # modify constraints not allowed
               "linear5", # modify constraints not allowed
               "linear6", # constraint set for l/q not allowed
               "linear7",  # VectorAffineFunction not supported.
               "linear8a", # Behavior in infeasible case doesn't match test.
               "linear8b", # Behavior in unbounded case doesn't match test.
               "linear8c", # Behavior in unbounded case doesn't match test.
               "linear10", # Interval not supported yet
               "linear10b", # Interval not supported yet
               "linear11", # Variable cannot be deleted
               "linear12", # Behavior in infeasible case doesn't match test.
               "linear14", # Variable cannot be deleted
               "linear15", # VectorAffineFunction not supported.
               ]
    MOIT.contlineartest(optimizer, config_no_duals,exclude)
end

@testset "MOI NLP tests" begin
    optimizer = MadNLP.Optimizer(print_level=MadNLP.ERROR)
    exclude = [
        "feasibility_sense_with_objective_and_no_hessian", # we need Hessians
        "feasibility_sense_with_no_objective_and_no_hessian", # we need Hessians
        "hs071_no_hessian", # we need Hessians
        "hs071_hessian_vector_product_test", # Hessian-vector product is needed
    ]
    MOIT.nlptest(optimizer,config,exclude)
end

@testset "Unit" begin
    bridged = MOIB.full_bridge_optimizer(MadNLP.Optimizer(print_level=MadNLP.ERROR),Float64)
    exclude = ["delete_variable", # Deleting not supported.
               "delete_variables", # Deleting not supported.
               "getvariable", # Variable names not supported.
               "solve_zero_one_with_bounds_1", # Variable names not supported.
               "solve_zero_one_with_bounds_2", # Variable names not supported.
               "solve_zero_one_with_bounds_3", # Variable names not supported.
               "getconstraint", # Constraint names not suported.
               "variablenames", # Variable names not supported.
               "solve_with_upperbound", # loadfromstring!
               "solve_with_lowerbound", # loadfromstring!
               "solve_integer_edge_cases", # loadfromstring!
               "solve_affine_lessthan", # loadfromstring!
               "solve_affine_greaterthan", # loadfromstring!
               "solve_affine_equalto", # loadfromstring!
               "solve_affine_interval", # loadfromstring!
               "get_objective_function", # Function getters not supported.
               "solve_constant_obj",  # loadfromstring!
               "solve_blank_obj", # loadfromstring!
               "solve_singlevariable_obj", # loadfromstring!
               "solve_objbound_edge_cases", # ObjectiveBound not supported.
               "solve_affine_deletion_edge_cases", # Deleting not supported.
               "solve_unbounded_model", # `NORM_LIMIT`
               "number_threads", # NumberOfThreads not supported
               "delete_nonnegative_variables", # get ConstraintFunction n/a.
               "update_dimension_nonnegative_variables", # get ConstraintFunction n/a.
               "delete_soc_variables", # VectorOfVar. in SOC not supported
               "solve_result_index", # DualObjectiveValue not supported
               "time_limit_sec", #time limit given as Flaot64?
               "solve_farkas_interval_lower",
               "solve_farkas_lessthan",
               "solve_farkas_interval_upper",
               "solve_farkas_greaterthan",
               "solve_farkas_variable_lessthan_max",
               "solve_farkas_variable_lessthan",
               "solve_farkas_equalto_lower",
               "solve_farkas_equalto_upper",
               "solve_qp_edge_cases"
               ]
    MOIT.unittest(bridged, config, exclude)
end

@testset "MOI QP/QCQP tests" begin
    optimizer = MadNLP.Optimizer(print_level=MadNLP.ERROR)
    qp_optimizer = MOIU.CachingOptimizer(MOIU.Model{Float64}(), optimizer)
    MOIT.qptest(qp_optimizer, config)
    exclude = ["qcp1", # VectorAffineFunction not supported.
              ]
    MOIT.qcptest(qp_optimizer, config_no_duals, exclude)
end
