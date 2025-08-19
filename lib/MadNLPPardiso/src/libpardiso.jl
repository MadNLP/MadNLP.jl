function pardisoinit(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libpardiso.pardisoinit(arg1::Ptr{Cvoid}, arg2::Ptr{Cint}, arg3::Ptr{Cint},
                                  arg4::Ptr{Cint}, arg5::Ptr{Cdouble},
                                  arg6::Ptr{Cint})::Cvoid
end

function pardiso(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12,
                 arg13, arg14, arg15, arg16, arg17)
    @ccall libpardiso.pardiso(arg1::Ptr{Cvoid}, arg2::Ptr{Cint}, arg3::Ptr{Cint},
                              arg4::Ptr{Cint}, arg5::Ptr{Cint}, arg6::Ptr{Cint},
                              arg7::Ptr{Cvoid}, arg8::Ptr{Cint}, arg9::Ptr{Cint},
                              arg10::Ptr{Cint}, arg11::Ptr{Cint}, arg12::Ptr{Cint},
                              arg13::Ptr{Cint}, arg14::Ptr{Cvoid}, arg15::Ptr{Cvoid},
                              arg16::Ptr{Cint}, arg17::Ptr{Cdouble})::Cvoid
end

function pardiso_chkmatrix(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libpardiso.pardiso_chkmatrix(arg1::Ptr{Cint}, arg2::Ptr{Cint},
                                        arg3::Ptr{Cdouble}, arg4::Ptr{Cint},
                                        arg5::Ptr{Cint}, arg6::Ptr{Cint})::Cvoid
end

function pardiso_chkvec(arg1, arg2, arg3, arg4)
    @ccall libpardiso.pardiso_chkvec(arg1::Ptr{Cint}, arg2::Ptr{Cint}, arg3::Ptr{Cdouble},
                                     arg4::Ptr{Cint})::Cvoid
end

function pardiso_printstats(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    @ccall libpardiso.pardiso_printstats(arg1::Ptr{Cint}, arg2::Ptr{Cint},
                                         arg3::Ptr{Cdouble}, arg4::Ptr{Cint},
                                         arg5::Ptr{Cint}, arg6::Ptr{Cint},
                                         arg7::Ptr{Cdouble}, arg8::Ptr{Cint})::Cvoid
end

function pardisoinit_z(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libpardiso.pardisoinit_z(arg1::Ptr{Cvoid}, arg2::Ptr{Cint}, arg3::Ptr{Cint},
                                    arg4::Ptr{Cint}, arg5::Ptr{Cdouble},
                                    arg6::Ptr{Cint})::Cvoid
end

function pardiso_z(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                   arg12, arg13, arg14, arg15, arg16, arg17)
    @ccall libpardiso.pardiso_z(arg1::Ptr{Cvoid}, arg2::Ptr{Cint}, arg3::Ptr{Cint},
                                arg4::Ptr{Cint}, arg5::Ptr{Cint}, arg6::Ptr{Cint},
                                arg7::Ptr{Cvoid}, arg8::Ptr{Cint}, arg9::Ptr{Cint},
                                arg10::Ptr{Cint}, arg11::Ptr{Cint}, arg12::Ptr{Cint},
                                arg13::Ptr{Cint}, arg14::Ptr{Cvoid}, arg15::Ptr{Cvoid},
                                arg16::Ptr{Cint}, arg17::Ptr{Cdouble})::Cvoid
end

function pardiso_chkmatrix_z(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libpardiso.pardiso_chkmatrix_z(arg1::Ptr{Cint}, arg2::Ptr{Cint},
                                          arg3::Ptr{Cvoid}, arg4::Ptr{Cint},
                                          arg5::Ptr{Cint}, arg6::Ptr{Cint})::Cvoid
end

function pardiso_chkvec_z(arg1, arg2, arg3, arg4)
    @ccall libpardiso.pardiso_chkvec_z(arg1::Ptr{Cint}, arg2::Ptr{Cint}, arg3::Ptr{Cvoid},
                                       arg4::Ptr{Cint})::Cvoid
end

function pardiso_printstats_z(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    @ccall libpardiso.pardiso_printstats_z(arg1::Ptr{Cint}, arg2::Ptr{Cint},
                                           arg3::Ptr{Cvoid}, arg4::Ptr{Cint},
                                           arg5::Ptr{Cint}, arg6::Ptr{Cint},
                                           arg7::Ptr{Cvoid}, arg8::Ptr{Cint})::Cvoid
end

function pardiso_get_schur_z(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libpardiso.pardiso_get_schur_z(arg1::Ptr{Cvoid}, arg2::Ptr{Cint},
                                          arg3::Ptr{Cint}, arg4::Ptr{Cint},
                                          arg5::Ptr{Cvoid}, arg6::Ptr{Cint},
                                          arg7::Ptr{Cint})::Cvoid
end

function pardisoinit_d(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libpardiso.pardisoinit_d(arg1::Ptr{Cvoid}, arg2::Ptr{Cint}, arg3::Ptr{Cint},
                                    arg4::Ptr{Cint}, arg5::Ptr{Cdouble},
                                    arg6::Ptr{Cint})::Cvoid
end

function pardiso_d(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                   arg12, arg13, arg14, arg15, arg16, arg17)
    @ccall libpardiso.pardiso_d(arg1::Ptr{Cvoid}, arg2::Ptr{Cint}, arg3::Ptr{Cint},
                                arg4::Ptr{Cint}, arg5::Ptr{Cint}, arg6::Ptr{Cint},
                                arg7::Ptr{Cvoid}, arg8::Ptr{Cint}, arg9::Ptr{Cint},
                                arg10::Ptr{Cint}, arg11::Ptr{Cint}, arg12::Ptr{Cint},
                                arg13::Ptr{Cint}, arg14::Ptr{Cvoid}, arg15::Ptr{Cvoid},
                                arg16::Ptr{Cint}, arg17::Ptr{Cdouble})::Cvoid
end

function pardiso_chkmatrix_d(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libpardiso.pardiso_chkmatrix_d(arg1::Ptr{Cint}, arg2::Ptr{Cint},
                                          arg3::Ptr{Cvoid}, arg4::Ptr{Cint},
                                          arg5::Ptr{Cint}, arg6::Ptr{Cint})::Cvoid
end

function pardiso_chkvec_d(arg1, arg2, arg3, arg4)
    @ccall libpardiso.pardiso_chkvec_d(arg1::Ptr{Cint}, arg2::Ptr{Cint}, arg3::Ptr{Cvoid},
                                       arg4::Ptr{Cint})::Cvoid
end

function pardiso_printstats_d(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    @ccall libpardiso.pardiso_printstats_d(arg1::Ptr{Cint}, arg2::Ptr{Cint},
                                           arg3::Ptr{Cvoid}, arg4::Ptr{Cint},
                                           arg5::Ptr{Cint}, arg6::Ptr{Cint},
                                           arg7::Ptr{Cvoid}, arg8::Ptr{Cint})::Cvoid
end

function pardiso_get_schur_d(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libpardiso.pardiso_get_schur_d(arg1::Ptr{Cvoid}, arg2::Ptr{Cint},
                                          arg3::Ptr{Cint}, arg4::Ptr{Cint},
                                          arg5::Ptr{Cvoid}, arg6::Ptr{Cint},
                                          arg7::Ptr{Cint})::Cvoid
end
