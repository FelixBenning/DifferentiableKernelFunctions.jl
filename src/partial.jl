const IndexType = Int # Union{Int,Base.AbstractCartesianIndex}

struct Partial{Order}
    indices::NTuple{Order,IndexType}
end

function Partial(indices::Integer...)
    return Partial{length(indices)}(indices)
end


## show helpers

### Fallbacks
compact_representation(p::Partial) = compact_representation(MIME"text/plain"(), p)
compact_representation(::MIME, p::Partial) = compact_representation(p)
detailed_representation(p::Partial) = """: Partial($(join(p.indices,",")))"""
detailed_representation(p::Partial{0}) = """: Partial() a zero order derivative"""

### text/plain
compact_representation(::MIME"text/plain", ::Partial{0}) = "id"
function compact_representation(::MIME"text/plain", p::Partial)
    tuple = Tuple(p.indices)
    lower_numbers = @. (n -> '₀' + n)(reverse(digits(tuple)))
    return join(["∂$(join(x))" for x in lower_numbers])
end


### text/html
compact_representation(::MIME"text/html", ::Partial{0}) = """<span class="text-muted" title="a zero order derivative">id</span>"""
function compact_representation(::MIME"text/html", p::Partial)
    return join(map(n -> "∂<sub>$(n)</sub>", Tuple(p.indices)), "")
end

### show

function Base.show(io::IO, p::Partial)
    print(io, compact_representation(p))
end

for T in [MIME"text/plain", MIME"text/html"]
    function Base.show(io::IO, mime::T, p::Partial)
        print(io, compact_representation(mime, p))
        get(io, :compact, false) && return
        print(io, detailed_representation(p))
    end
end

const DiffPt{T} = Tuple{T, Partial}

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
