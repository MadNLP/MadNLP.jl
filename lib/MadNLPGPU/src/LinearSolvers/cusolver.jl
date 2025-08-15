introduce(M::LapackGPUSolver{T,V}) where {T,V<:CuVector} = "cuSOLVER v$(CUSOLVER.version()) -- ($(M.opt.lapack_algorithm))"

for (potrf, potrf_buffer, potrs, T) in
    ((:cusolverDnDpotrf, :cusolverDnDpotrf_bufferSize, :cusolverDnDpotrs, :Float64),
     (:cusolverDnSpotrf, :cusolverDnSpotrf_bufferSize, :cusolverDnSpotrs, :Float32))
    @eval begin
        function setup_cholesky!(M::LapackGPUSolver{$T,V}) where {V<:CuVector}
            potrf_lwork_gpu = Ref{Cint}(0)
            CUSOLVER.$potrf_buffer(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Cint(M.n),
                M.fact,
                Cint(M.n),
                potrf_lwork_gpu,
            )
            M.lwork_gpu = potrf_lwork_gpu[]
            resize!(M.work_gpu, M.lwork_gpu)
            return M
        end

        function factorize_cholesky!(M::LapackGPUSolver{$T,V}) where {V<:CuVector}
            CUSOLVER.$potrf(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.work_gpu,
                Cint(M.lwork_gpu),
                M.info,
            )
            return M
        end

        function solve_cholesky!(M::LapackGPUSolver{$T,V}, x::V) where {V<:CuVector}
            CUSOLVER.$potrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Cint(M.n),
                one(Cint),
                M.fact,
                Cint(M.n),
                x,
                Cint(M.n),
                M.info,
            )
            return x
        end
    end
end

for (sytrf_buffer, sytrf, sytrs_buffer, sytrs, T) in
    ((:cusolverDnDsytrf_bufferSize, :cusolverDnDsytrf, :cusolverDnXsytrs_bufferSize, :cusolverDnXsytrs, :Float64),
     (:cusolverDnSsytrf_bufferSize, :cusolverDnSsytrf, :cusolverDnXsytrs_bufferSize, :cusolverDnXsytrs, :Float32))
    @eval begin
        function setup_bunchkaufman!(M::LapackGPUSolver{$T,V}) where {V<:CuVector}
            resize!(M.ipiv, M.n)
            resize!(M.ipiv64, M.n)
            sytrf_lwork_gpu = Ref{Cint}(0)
            CUSOLVER.$sytrf_buffer(
                dense_handle(),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                sytrf_lwork_gpu,
            )
            sytrs_lwork_cpu = Ref{Csize_t}(0)  # size in bytes!
            sytrs_lwork_gpu = Ref{Csize_t}(0)  # size in bytes!
            CUSOLVER.$sytrs_buffer(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                M.n,
                one(Int64),
                $T,
                M.fact,
                M.n,
                M.ipiv64,
                $T,
                M.tau, # We can use any vector of the same length as the solution, which is M.n
                M.n,
                sytrs_lwork_gpu,
                sytrs_lwork_cpu,
            )
            M.lwork_cpu = sytrs_lwork_cpu[]
            M.lwork_gpu = max(sytrf_lwork_gpu[], sytrs_lwork_gpu[] ÷ sizeof($T))
            resize!(M.work_cpu, M.lwork_cpu)
            resize!(M.work_gpu, M.lwork_gpu)
            return M
        end

        function factorize_bunchkaufman!(M::LapackGPUSolver{$T,V}) where {V<:CuVector}
            CUSOLVER.$sytrf(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.ipiv,
                M.work_gpu,
                Cint(M.lwork_gpu),
                M.info,
            )
            return M
        end

        function solve_bunchkaufman!(M::LapackGPUSolver{$T,V}, x::V) where {V<:CuVector}
            copyto!(M.ipiv64, M.ipiv)  # No workaround possible until NVIDIA implements cusolverDnXsytrf
            CUSOLVER.$sytrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                M.n,
                one(Int64),
                $T,
                M.fact,
                M.n,
                M.ipiv64,
                $T,
                x,
                M.n,
                M.work_gpu,
                Csize_t(M.lwork_gpu),
                M.work_cpu,
                Csize_t(M.lwork_cpu),
                M.info,
            )
            return x
        end
    end
end

for (getrf, getrf_buffer, getrs, T) in
    ((:cusolverDnDgetrf, :cusolverDnDgetrf_bufferSize, :cusolverDnDgetrs, :Float64),
     (:cusolverDnSgetrf, :cusolverDnSgetrf_bufferSize, :cusolverDnSgetrs, :Float32))
    @eval begin
        function setup_lu!(M::LapackGPUSolver{$T,V}) where {V<:CuVector}
            resize!(M.ipiv, M.n)
            getrf_lwork_gpu = Ref{Cint}(0)
            CUSOLVER.$getrf_buffer(
                dense_handle(),
                Cint(M.n),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                getrf_lwork_gpu,
            )
            M.lwork_gpu = getrf_lwork_gpu[]
            resize!(M.work_gpu, M.lwork_gpu)
            return M
        end

        function factorize_lu!(M::LapackGPUSolver{$T,V}) where {V<:CuVector}
            CUSOLVER.$getrf(
                dense_handle(),
                Cint(M.n),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.work_gpu,
                M.ipiv,
                M.info,
            )
            return M
        end

        function solve_lu!(M::LapackGPUSolver{$T,V}, x::V) where {V<:CuVector}
            CUSOLVER.$getrs(
                dense_handle(),
                CUBLAS_OP_N,
                Cint(M.n),
                one(Cint),
                M.fact,
                Cint(M.n),
                M.ipiv,
                x,
                Cint(M.n),
                M.info,
            )
            return x
        end
    end
end

for (geqrf, geqrf_buffer, ormqr, ormqr_buffer, trsm, T) in
    ((:cusolverDnDgeqrf, :cusolverDnDgeqrf_bufferSize, :cusolverDnDormqr, :cusolverDnDormqr_bufferSize, :cublasDtrsm_v2_64, :Float64),
     (:cusolverDnSgeqrf, :cusolverDnSgeqrf_bufferSize, :cusolverDnSormqr, :cusolverDnSormqr_bufferSize, :cublasStrsm_v2_64, :Float32))
    @eval begin
        function setup_qr!(M::LapackGPUSolver{$T,V}) where {V<:CuVector}
            resize!(M.tau, M.n)
            geqrf_lwork_gpu = Ref{Cint}(0)
            CUSOLVER.$geqrf_buffer(
                dense_handle(),
                Cint(M.n),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                geqrf_lwork_gpu,
            )
            ormqr_lwork_gpu = Ref{Cint}(0)
            CUSOLVER.$ormqr_buffer(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Cint(M.n),
                one(Cint),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.tau,
                M.tau,  # We can use any vector of the same length as the solution, which is M.n
                Cint(M.n),
                ormqr_lwork_gpu,
            )
            M.lwork_gpu = max(geqrf_lwork_gpu[], ormqr_lwork_gpu[])
            resize!(M.work_gpu, M.lwork_gpu)
            return M
        end

        function factorize_qr!(M::LapackGPUSolver{$T,V}) where {V<:CuVector}
            CUSOLVER.$geqrf(
                dense_handle(),
                Cint(M.n),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.tau,
                M.work_gpu,
                Cint(M.lwork_gpu),
                M.info,
            )
            return M
        end

        function solve_qr!(M::LapackGPUSolver{$T,V}, x::V) where {V<:CuVector}
            CUSOLVER.$ormqr(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Cint(M.n),
                one(Cint),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.tau,
                x,
                Cint(M.n),
                M.work_gpu,
                Cint(M.lwork_gpu),
                M.info,
            )
            CUBLAS.$trsm(
                handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                M.n,
                one(Int64),
                one($T),
                M.fact,
                M.n,
                x,
                M.n,
            )
            return x
        end
    end
end
