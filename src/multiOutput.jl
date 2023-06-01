
using MappedArrays: mappedarray, ReadonlyMappedArray

function ensure_all_linear_indexed(vecs::T) where {T<:Tuple}
    linear_indexed = ntuple(
        n -> hasmethod(Base.getindex, (fieldtype(T, n), Int)),
        Base._counttuple(T)
    )
    all(linear_indexed) || throw(ArgumentError(
        "$(vecs[findfirst(x->!x, linear_indexed)]) cannot be linearly accessed. All inputs need to implement `Base.getindex(::T, ::Int)`"
    ))
end

struct ProductArray{T<:Tuple,N,Eltype} <: AbstractArray{Eltype,N}
    prodIt::Iterators.ProductIterator{T}
    ProductArray(t::T) where T = begin
        ensure_all_linear_indexed(t)
        prodIt = Iterators.ProductIterator(t)
        new{T,ndims(prodIt),eltype(Iterators.ProductIterator{T})}(prodIt)
    end
end

# wrap ProductIterator
function Base.IteratorSize(::Type{ProductArray{T,N, Eltype}}) where {T,N,Eltype}
    Base.IteratorSize(Iterators.ProductIterator{T})
end
Base.size(p::ProductArray) = size(p.prodIt)
Base.axes(p::ProductArray) = axes(p.prodIt)
Base.ndims(::ProductArray{T,N}) where {T,N} = N
Base.length(p::ProductArray) = length(p.prodIt)
function Base.IteratorEltype(::Type{<:ProductArray{T}}) where {T}
    Base.IteratorEltype(Iterators.ProductIterator{T})
end 
Base.eltype(::Type{ProductArray{T,N,Eltype}}) where {T,N,Eltype} = Eltype
Base.iterate(p::ProductArray) = iterate(p.prodIt)
Base.iterate(p::ProductArray, state) = iterate(p.prodIt, state)

Base.last(p::ProductArray) = last(p.prodIt)

# implement private _getindex for ProductIterator

function _getindex(prod::Iterators.ProductIterator, indices::Int...)
    return _prod_getindex(prod.iterators, indices...)
end
_prod_getindex(::Tuple{}) = ()
function _prod_getindex(p_vecs::Tuple, indices::Int...)
    v = first(p_vecs)
    n = ndims(v)
    return (
        v[indices[1:n]...],
        _prod_getindex(Base.tail(p_vecs), indices[n+1:end]...)...
    )
end

# apply this to ProductArray
Base.getindex(p::ProductArray{T,N}, indices::Vararg{Int,N}) where {T,N} = _getindex(p.prodIt, indices...)

"""
    lazy_product(vectors...)

The output is a lazy form of
```julia
collect(Iterators.product(vectors...))
```
i.e. it is an AbstractArray in contrast to `Iterators.product(vectors...)`. So
is accessible with `getindex` and gets default Array implementations for free.
In particular it can be passed to `Base.PermutedDimsArray`` for lazy permutation
and `vec()` to obtain a lazy `Base.ReshapedArray`.
"""
lazy_product(vectors...) = ProductArray(vectors)

"""
    lazy_flatten(vectors...)

The output is a lazy form of
```julia
collect(Iterators.flatten(vectors...))
```
i.e. it is an AbstractArray in contrast to `Iterators.flatten(vectors...)`. So
is accessible with `getindex` and gets default Array implementations for free.
In particular it can be passed to `Base.PermutedDimsArray`` for lazy permutation
and `vec()` to obtain a lazy `Base.ReshapedArray`.
"""
function lazy_flatten(vectors...)
    ensure_all_linear_indexed(vectors)
    lengths = cumsum(length.(vectors))
    return mappedarray(1:lengths[end]) do idx
        # this is not efficient for iteration - maybe go with LazyArrays.jl -> Vcat instead.
        v_idx = searchsortedfirst(lengths, idx)
        return vectors[v_idx][idx-get(lengths, v_idx - 1, 0)]
    end
end


function multi_out_byFeatures(positions, out_dims)
    return vec(PermutedDimsArray(lazy_product(positions, 1:out_dims), (2, 1)))
end
function multi_out_byOutput(positions, out_dims)
    return vec(lazy_product(positions, 1:out_dims))
end

