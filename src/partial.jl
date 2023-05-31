const IndexType = Int # Union{Int,Base.AbstractCartesianIndex}

struct Partial{Order}
    indices::NTuple{Order,IndexType}
end

function Partial(indices::Integer...)
    return Partial{length(indices)}(indices)
end

const DiffPt{T} = Tuple{T, Partial}

compact_string_representation(::Partial{0}) = print(io, "id")
function compact_string_representation(p::Partial)
    tuple = Tuple(p.indices)
    lower_numbers = @. (n -> '₀' + n)(reverse(digits(tuple)))
    return join(["∂$(join(x))" for x in lower_numbers])
end

function Base.show(io::IO, ::MIME"text/plain", p::Partial)
    if get(io, :compact, false)
        print(io, "Partial($(Tuple(p.indices)))")
    else
        print(io, compact_string_representation(p))
    end
end

function Base.show(io::IO, ::MIME"text/html", p::Partial)
    tuple = Tuple(p.indices)
    if get(io, :compact, false)
        print(io, join(map(n -> "∂<sub>$(n)</sub>", tuple), ""))
    else
        print(io, compact_string_representation(p))
    end
end


"""
    tangentCurve(x₀, i::IndexType)
returns the function (t ↦ x₀ + teᵢ) where eᵢ is the unit vector at index i
"""
function tangentCurve(x0::AbstractArray, idx::IndexType)
    return t -> begin
        x = similar(x0, promote_type(eltype(x0), typeof(t)))
        copyto!(x, x0)
        x[idx] += t
        return x
    end
end
function tangentCurve(x0::Number, ::IndexType)
    return t -> x0 + t
end

partial(func) = func
function partial(func, idx::IndexType)
    return x -> FD.derivative(func ∘ tangentCurve(x, idx), 0)
end
function partial(func, partials::IndexType...)
    idx, state = iterate(partials)
    return partial(
        x -> FD.derivative(func ∘ tangentCurve(x, idx), 0), Base.rest(partials, state)...
    )
end

"""
Take the partial derivative of a function with two dim-dimensional inputs,
i.e. 2*dim dimensional input
"""
function partial(
    k, partials_x::Tuple{Vararg{T}}, partials_y::Tuple{Vararg{T}}
) where {T<:IndexType}
    local f(x, y) = partial(t -> k(t, y), partials_x...)(x)
    return (x, y) -> partial(t -> f(x, t), partials_y...)(y)
end
