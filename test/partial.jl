@testset "Partial" begin
    @test partial.(1:10) == DKF.gradient(10)
    @test partial.(productArray(1:10, 1:10)) == DKF.hessian(10)
    @test DKF.gradient(10) isa AbstractArray{DKF.Partial{1,Tuple{Int}},1}
    @test DKF.gradient(2) == DKF.fullderivative(Val(1), 2)
    @test size(DKF.gradient(CartesianIndices((4,3)))) == (4,3)
    @test ndims(DKF.hessian(CartesianIndices((2,2,2)))) == 6
end