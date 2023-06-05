@testset "diffKernel" begin
    @testset "smoke test" begin
        k = EnableDiff(MaternKernel())
        k2 = MaternKernel()
        @test k(1, 1) == k2(1, 1)
        k(1, (1, partial(1, 1))) # Cov(Z(x), ∂₁∂₁Z(y)) where x=1, y=1
        k(([1], partial(1)), [2]) # Cov(∂₁Z(x), Z(y)) where x=[1], y=[2]
    end

    @testset "Sanity Checks with $k1" for k1 in [
        SEKernel(),
        MaternKernel(ν=5),
        RationalQuadraticKernel(),
        SEKernel() + RationalQuadraticKernel()
    ]
        k = EnableDiff(k1)
        for x in [0, 1, -1, 42]
            # correlation with self should be positive 
            ## This fails for Matern and RationalQuadraticKernel
            # because its implementation branches on x == y resulting in a zero derivative
            # (cf. https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues/517)
            @test k((x, partial(1)), (x, partial(1))) > 0

            # the slope should be positively correlated with a point further down
            @test k(
                (x, partial(1)), # slope
                x + 1e-2, # point further down
            ) > 0

            @testset "Stationary Tests" begin
                @test k((x, partial(1)), x) == 0 # expect Cov(∂Z(x) , Z(x)) == 0

                @testset "Isotropic Tests" begin
                    @test k(([1, 2], partial(1)), ([1, 2], partial(2))) == 0 # cross covariance should be zero
                end
            end
        end
    end
end
