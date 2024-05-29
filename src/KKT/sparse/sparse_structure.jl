"""
    SparseKKTSystem{T, VT, MT, QN} <: AbstractReducedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractReducedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractReducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}
    # Hessian
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}
    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end


"""
    SparseUnreducedKKTSystem{T, VT, MT, QN} <: AbstractUnreducedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractUnreducedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseUnreducedKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractUnreducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    l_lower_aug::VT
    u_lower_aug::VT

    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}

    # Hessian
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}

    # LinearSolver
    linear_solver::LS

    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

"""
    SparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractCondensedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseCondensedKKTSystem{T, VT, MT, QN, LS, VI, VI32, VTu1, VTu2, EXT} <: AbstractCondensedKKTSystem{T, VT, MT, QN}
    # Hessian
    hess::VT
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # Jacobian
    jac::VT
    jt_coo::SparseMatrixCOO{T,Int32,VT, VI32}
    jt_csc::MT
    jt_csc_map::Union{Nothing, VI}

    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # buffer
    buffer::VT
    buffer2::VT

    # Augmented system
    aug_com::MT

    # slack diagonal buffer
    diag_buffer::VT
    dptr::VTu1
    hptr::VTu1
    jptr::VTu2

    # LinearSolver
    linear_solver::LS

    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # extra
    ext::EXT
end

# Template to dispatch on sparse representation
const AbstractSparseKKTSystem{T, VT, MT, QN} = Union{
    SparseKKTSystem{T, VT, MT, QN},
    SparseCondensedKKTSystem{T, VT, MT, QN},
    SparseUnreducedKKTSystem{T, VT, MT, QN},
}

"""
    ScaledSparseKKTSystem{T, VT, MT, QN} <: AbstractReducedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractReducedKKTSystem`](@ref) in sparse COO format.

"""
struct ScaledSparseKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractReducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}
    # Hessian
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}
    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end


"""
    ScaledSparseUnreducedKKTSystem{T, VT, MT, QN} <: AbstractUnreducedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractUnreducedKKTSystem`](@ref) in sparse COO format.

"""
struct ScaledSparseUnreducedKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractUnreducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    l_lower_aug::VT
    u_lower_aug::VT

    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}

    # Hessian
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}

    # LinearSolver
    linear_solver::LS

    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

"""
    ScaledSparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractCondensedKKTSystem`](@ref) in sparse COO format.

"""
struct ScaledSparseCondensedKKTSystem{T, VT, MT, QN, LS, VI, VI32, VTu1, VTu2, EXT} <: AbstractCondensedKKTSystem{T, VT, MT, QN}
    # Hessian
    hess::VT
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # Jacobian
    jac::VT
    jt_coo::SparseMatrixCOO{T,Int32,VT, VI32}
    jt_csc::MT
    jt_csc_map::Union{Nothing, VI}

    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # buffer
    buffer::VT
    buffer2::VT

    # Augmented system
    aug_com::MT

    # slack diagonal buffer
    diag_buffer::VT
    dptr::VTu1
    hptr::VTu1
    jptr::VTu2

    # LinearSolver
    linear_solver::LS

    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # extra
    ext::EXT
end

# Template to dispatch on sparse representation
const AbstractScaledSparseKKTSystem{T, VT, MT, QN} = Union{
    ScaledSparseKKTSystem{T, VT, MT, QN},
    ScaledSparseCondensedKKTSystem{T, VT, MT, QN},
    ScaledSparseUnreducedKKTSystem{T, VT, MT, QN},
}
