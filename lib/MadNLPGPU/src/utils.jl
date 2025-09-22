# ordering
@enum ORDERING begin
    DEFAULT_ORDERING = 0
    METIS_ORDERING = 1
    AMD_ORDERING = 2
    USER_ORDERING = 3
    SYMAMD_ORDERING = 4
    COLAMD_ORDERING = 5
end

# Local function to move data on the device.
gpu_transfer!(x::AbstractArray, y::AbstractArray) = copyto!(x, y)
