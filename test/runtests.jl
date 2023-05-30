import KernelFunctions as KF
using DifferentiableKernelFunctions: DifferentiableKernelFunctions as DKF
using Test

@testset "DifferentiableKernelFunctions.jl" begin
    @testset "custom product identical to Base.product" begin
        @test DKF.product(1:3, 4:10) == collect(Base.product(1:3, 4:10))
    end
end

