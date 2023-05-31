
using MappedArrays: mappedarray

function ensure_all_linear_indexed(vecs::T) where {T<:Tuple}
    linear_indexed = ntuple(
        n -> Base.IndexStyle(fieldtype(T,n)) === IndexLinear(),
        Base._counttuple(T)
    )
    all(linear_indexed) || throw(ArgumentError(
        "$(vecs[findfirst(x->!x, linear_indexed)]) cannot be linearly accessed. All inputs need to implement `Base.getindex(::T, ::Int)`"
    ))
end

struct ProductArray
    arrays
end

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
function lazy_product(vectors...)
    ensure_all_linear_indexed(vectors)
    indices = CartesianIndices(ntuple(n -> axes(vec(vectors[n]), 1), length(vectors)))
    return mappedarray(indices) do idx
        return ntuple(n -> vec(vectors[n])[idx[n]], length(vectors))
    end
end

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
        return vectors[v_idx][idx - get(lengths, v_idx-1, 0)]
    end
end


function multi_out_byFeatures(positions, out_dims)
    return vec(PermutedDimsArray(lazy_product(positions, 1:out_dims), (2, 1)))
end
function multi_out_byOutput(positions, out_dims)
    return vec(lazy_product(positions, 1:out_dims))
end

