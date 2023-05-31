@testset "multiOutput" begin
	@testset "custom product identical to Base.product" begin
		@test DKF.lazy_product(1:3, 4:10) == collect(Base.product(1:3, 4:10))
	end
end