const IndexType = Union{Int,Base.AbstractCartesianIndex}

struct Partial{Order,T<:IndexType}
    indices::NTuple{Order,T}
end

# TODO: this is not ideal... how does NTuple{0,Int} <: Tuple{} work??
Partial() = Partial{0,Int}(())
function Partial(indices::Integer...)
    return Partial{length(indices),Int}(indices)
end
function Partial(indices::Base.AbstractCartesianIndex...)
    return Partial{length(indices),Base.AbstractCartesianIndex}(indices)
end
partial(indices...) = Partial(indices...)

## show helpers

lower_digits(n::Integer) = join(reverse(digits(n)) .+ '₀')
lower_digits(idx::Base.AbstractCartesianIndex) = join(map(lower_digits, Tuple(idx)), ",")

### Fallbacks
compact_representation(p::Partial) = compact_representation(MIME"text/plain"(), p)
compact_representation(::MIME, p::Partial) = compact_representation(p)
detailed_representation(p::Partial) = """: Partial($(join(p.indices,",")))"""
detailed_representation(p::Partial{0}) = """: Partial() a zero order derivative"""

### text/plain
compact_representation(::MIME"text/plain", ::Partial{0}) = "id"
function compact_representation(::MIME"text/plain", p::Partial)
    lower_numbers = map(lower_digits, p.indices)
    return join(["∂$(x)" for x in lower_numbers])
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

const DiffPt{T} = Tuple{T,Partial}

gradient(dim::Integer) = mappedarray(partial, Base.OneTo(dim))
hessian(dim::Integer) = mappedarray(partial, lazy_product(Base.OneTo(dim), Base.OneTo(dim)))
fullderivative(order::Integer,dim::Integer) = mappedarray(partial, lazy_product(ntuple(_->Base.OneTo(dim), order)...))

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

apply_partial(func) = func
function apply_partial(func, idx::IndexType)
    return x -> FD.derivative(func ∘ tangentCurve(x, idx), 0)
end
function apply_partial(func, partials::IndexType...)
    idx, state = iterate(partials)
    return apply_partial(
        x -> FD.derivative(func ∘ tangentCurve(x, idx), 0), Base.rest(partials, state)...
    )
end

"""
Take the partial derivative of a function with two dim-dimensional inputs,
i.e. 2*dim dimensional input
"""
function apply_partial(
    k, partials_x::Tuple{Vararg{T}}, partials_y::Tuple{Vararg{T}}
) where {T<:IndexType}
    local f(x, y) = apply_partial(t -> k(t, y), partials_x...)(x)
    return (x, y) -> apply_partial(t -> f(x, t), partials_y...)(y)
end
