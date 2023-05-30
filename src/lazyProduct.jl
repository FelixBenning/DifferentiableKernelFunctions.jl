module Lazy

using MappedArrays: mappedarray

function ensure_all_linear_indexed(vecs::T) where {T<:Tuple}
    linear_indexed = ntuple(
        n -> hasmethod(Base.getindex, (fieldtype(T, n), Int)),
        Base._counttuple(T)
    )
    all(linear_indexed) || throw(ArgumentError(
        "$(vecs[findfirst(x->!x, linear_indexed)]) cannot be linearly accessed. All inputs need to implement `Base.getindex(::T, ::Int)`"
    ))
end

function product(vectors...)
    ensure_all_linear_indexed(vectors)
    indices = CartesianIndices(ntuple(n -> axes(vec(vectors[n]), 1), length(vectors)))
    return mappedarray(indices) do idx
        return ntuple(n -> vec(vectors[n])[idx[n]], length(vectors))
    end
end


function multi_out_byFeatures(positions, out_dims)
    return vec(PermutedDimsArray(product(positions, 1:out_dims), (2, 1)))
end
function multi_out_byOutput(positions, out_dims)
    return vec(product(positions, 1:out_dims))
end

end # module Lazy