# MadNLP.jl.
# Created by Sungho Shin (sungho.shin@wisc.edu)

using Cxx

function NonlinearProgram(prob::Cxx.CxxCore.CppPtr)
    dummyInt32=Ptr{Int32}(C_NULL)
    dummyFloat64=Ptr{Float64}(C_NULL)

    n=Ref{Int32}(0)
    m=Ref{Int32}(0)
    n_jac=Ref{Int32}(0)
    n_hess=Ref{Int32}(0)
    index_style=Ref(@cxx TNLP::C_STYLE)
    @cxx prob->get_nlp_info(n,m,n_jac,n_hess,index_style)

    xl = Array{Float64,1}(undef,n.x)
    xu = Array{Float64,1}(undef,n.x)
    gl = Array{Float64,1}(undef,m.x)
    gu = Array{Float64,1}(undef,m.x)

    @cxx prob->get_bounds_info(n.x,pointer(xl),pointer(xu),m.x,pointer(gl),pointer(gu))
    
    x = zeros(n.x)
    zl= zeros(n.x)
    zu= zeros(n.x)
    l = zeros(m.x)

    @cxx prob->get_starting_point(n.x,true,pointer(x),
                                  true,pointer(zl),pointer(zu),
                                  m.x,true,pointer(l))

    objvalref = Ref{Float64}(0.)

    function obj(x::StridedArray{Float64})
        @cxx prob->eval_f(n.x,pointer(x),true,objvalref)
        return objvalref.x
    end

    function obj_grad!(f::StridedArray{Float64},x::StridedArray{Float64})
        @cxx prob->eval_grad_f(n.x,pointer(x),true,pointer(f))
    end

    function con!(c,x)
        @cxx prob->eval_g(n.x,pointer(x),true,m.x,pointer(c))
    end

    function con_jac!(jac,x)
        @cxx prob->eval_jac_g(n.x,pointer(x),true,m.x,n_jac.x,dummyInt32,dummyInt32,pointer(jac))
    end

    function lag_hess!(hess,x,l,sig)
        @cxx prob->eval_h(n.x,pointer(x),true,
                          sig,m.x,pointer(l),true,
                          n_hess.x,dummyInt32,dummyInt32,
                          pointer(hess))
    end
    function jac_sparsity(I,J)
        @cxx prob->eval_jac_g(n.x,dummyFloat64,false,m.x,n_jac.x,
                              pointer(I),pointer(J),dummyFloat64)
        (index_style.x == @cxx TNLP::C_STYLE) && (I.+=1;J.+=1)
        return I,J
    end
    function hess_sparsity(I,J)
        @cxx prob->eval_h(n.x,dummyFloat64,false,
                          1.,m.x,dummyFloat64,false,
                          n_hess.x,pointer(I),pointer(J),
                          dummyFloat64)
        (index_style.x == @cxx TNLP::C_STYLE) && (I.+=1;J.+=1)
        return I,J
    end
    return ScalaOpt.NonlinearProgram(n[],m[],n_hess.x,n_jac.x,
                                     0.,x,l,zl,zu,
                                     xl,xu,gl,gu,
                                     obj,obj_grad!,con!,con_jac!,lag_hess!,
                                     hess_sparsity,jac_sparsity,
                                     false,false,
                                     :Initial)
end

function optimize!(prob::Cxx.CxxCore.CppPtr;kwargs...)
    NLP = NonlinearProgram(prob)
    IPS = ScalaOpt.InteriorPointSolver(NLP;kwargs...)
    ScalaOpt.interior_point_optimize!(IPS)
    return IPS
end
