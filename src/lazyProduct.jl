module Lazy
# Inspired by Iterators.ProductIterator
function _eltype_iterator_tuple(::Type{T}) where {T<:Tuple}
    return Tuple{ntuple(n -> eltype(fieldtype(T, n)), Base._counttuple(T))...}
end

struct ProductArray{T<:Tuple,Eltype,N} <: AbstractArray{Eltype,N}
    vectors::T
    ProductArray(vectors::T) where {T} = new{T,_eltype_iterator_tuple(T),Base._counttuple(T)}(vectors)
end

struct AllLinearIndexed end
struct GeneralIterators end

function _all_linear_indexed(::T) where {T<:Tuple}
    all(ntuple(
        n -> hasmethod(Base.getindex, (fieldtype(T, n), Int)),
        Base._counttuple(T)
    )) && return AllLinearIndexed()
    return GeneralIterators()
end
product(iterators...) = _product(_all_linear_indexed(iterators), iterators)
_product(::AllLinearIndexed, vectors) = ProductArray(vectors)
_product(::GeneralIterators, vectors) = Base.ProductIterator(vectors)



Base.size(prod::ProductArray) = _prod_size(prod.vectors)
_prod_size(::Tuple{}) = ()
_prod_size(t::Tuple) = (length(t[1]), _prod_size(Base.tail(t))...)

function Base.getindex(prod::ProductArray{T,Eltype,N}, indices::Vararg{Int,N}) where {T,Eltype,N}
    return _prod_getindex(prod.vectors, indices...)
end
_prod_getindex(::Tuple{}) = ()
function _prod_getindex(p_vecs::Tuple, indices...)
    return (
        first(p_vecs)[first(indices)],
        _prod_getindex(Base.tail(p_vecs), Base.tail(indices)...)...
    )
end

const PermutedProduct = Base.PermutedDimsArray{NTuple{N,T},N,perm,iperm,ProductArray{T,N}} where {T,N,perm,iperm}
end