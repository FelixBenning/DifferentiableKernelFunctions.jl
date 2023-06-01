@testset "multiOutput" begin
	@testset "ProductArray identical to collected Base.product" begin
		@test DKF.lazy_product(1:3, 4:10) == collect(Base.product(1:3, 4:10))
		v1 = rand(3, 2)
		v2 = [:a, :b]
		@test DKF.lazy_product(v1, v2) == collect(Base.product(v1, v2))

		v = (v1, v2)
		p = DKF.lazy_product(v...)

		@testset "Test Type Functions" begin
			@test Base.IteratorSize(typeof(p)) == Base.IteratorSize(typeof(p.prodIt))
			@test Base.IteratorEltype(typeof(p)) == Base.IteratorEltype(typeof(p.prodIt))
			@test Base.eltype(typeof(p)) == Base.eltype(typeof(p.prodIt))
		end

		c = collect(Base.product(v...))
		@testset "Test Entries and Access" begin
			@test p == c # same shape and entries
			cart_idx = map(first, axes(p))
			@test p[cart_idx...] == c[cart_idx...] # cartesian access
			@test p[1] == c[1] # linear access
			@test p[:] == c[:] # colon access
			@test p[1:lastindex(p)] == vec(c) # range access
			@test reverse(p) == reverse(c)
			@test last(p) == p[map(last, axes(p))...]
		end

		@testset "Sanity Checks" begin
			@test axes(p) == axes(c)
			@test size(p) == map(length, axes(p))
			@test length(p) == reduce(*, size(p))
			@test axes(p, 1) == axes(p)[1]
		end
	end
end