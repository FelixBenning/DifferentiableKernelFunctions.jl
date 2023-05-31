@testset "multiOutput" begin
	@testset "custom product identical to Base.product" begin
		@test DKF.lazy_product(1:3, 4:10) == collect(Base.product(1:3, 4:10))
		v1 = rand(3, 2)
		v2 = [:a, :b]
		@test DKF.lazy_product(v1, v2) == collect(Base.product(v1, v2)) skip=true 
	end
end