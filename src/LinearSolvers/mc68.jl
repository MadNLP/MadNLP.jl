# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module Mc68

import ..MadNLP: @kwdef, libhsl

@kwdef mutable struct Control
    f_array_in::Cint = 0
    f_array_out::Cint = 0
    min_l_workspace::Clong = 0
    lp::Cint = 0
    wp::Cint = 0
    mp::Cint = 0
    nemin::Cint = 0
    print_level::Cint = 0
    row_full_thresh::Cint = 0
    row_search::Cint = 0
end

@kwdef mutable struct Info
    flag::Cint = 0
    iostat::Cint = 0
    stat::Cint = 0
    out_range::Cint = 0
    duplicate::Cint = 0
    n_compressions::Cint = 0
    n_zero_eigs::Cint = 0
    l_workspace::Clong = 0
    zb01_info::Cint = 0
    n_dense_rows::Cint = 0
end

function get_mc68_default_control()
    control = Control(0,0,0,0,0,0,0,0,0,0)
    mc68_default_control_i(control)
    return control
end

mc68_default_control_i(control::Control) = ccall((:mc68_default_control_i,libhsl),
                                                 Nothing,
                                                 (Ref{Control},),
                                                 control)

mc68_order_i(ord::Int32,n::Int32,ptr::Array{Int32,1},row::Array{Int32,1},
             perm::Array{Int32,1},control::Control,info::Info) = ccall(
                 (:mc68_order_i,libhsl),
                 Nothing,
                 (Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ref{Control},Ref{Info}),
                 ord,n,ptr,row,perm,control,info)

end #module
