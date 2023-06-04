const IndexType = Union{Int,Base.AbstractCartesianIndex}
struct Partial{Order,T<:Tuple{Vararg{IndexType,Order}}}
    indices::T
end

Partial(::Tuple{}) = Partial{0,Tuple{}}(())
function Partial(indices::Tuple{Vararg{T}}) where {T<:IndexType}
    Ord = length(indices)
    return Partial{Ord,NTuple{Ord,T}}(indices)
end
partial(indices::Tuple{Vararg{T}}) where {T<:IndexType} = Partial(indices)
partial(indices::IndexType...) = Partial(indices)

## show helpers

lower_digits(n::Integer) = join(reverse(digits(n)) .+ '₀')
lower_digits(idx::Base.AbstractCartesianIndex) = join(map(lower_digits, Tuple(idx)), ",")

### Fallbacks
compact_representation(p::Partial) = compact_representation(MIME"text/plain"(), p)
compact_representation(::MIME, p::Partial) = compact_representation(p)
detailed_representation(p::Partial) = """: partial($(join(p.indices,",")))"""
detailed_representation(p::Partial{0,Tuple{}}) = """: partial() a zero order derivative"""

### text/plain
compact_representation(::MIME"text/plain", ::Partial{0,Tuple{}}) = "id"
function compact_representation(::MIME"text/plain", p::Partial)
    lower_numbers = map(lower_digits, p.indices)
    return join(["∂$(x)" for x in lower_numbers])
end

### text/html
compact_representation(::MIME"text/html", ::Partial{0,Tuple{}}) = """<span class="text-muted" title="a zero order derivative">id</span>"""
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

function fullderivative(::Val{order}, input_indices::AbstractVector{Int}) where {order}
    return mappedarray(partial, productArray(ntuple(_ -> input_indices, Val{order}())...))
end
fullderivative(::Val{order}, dim::Integer) where {order} = fullderivative(Val{order}(), Base.OneTo(dim))
function fullderivative(::Val{order}, input_indices::AbstractArray{T,N}) where {order,N,T<:Base.AbstractCartesianIndex{N}}
    return mappedarray(partial, productArray(ntuple(_ -> input_indices, Val{order}())...))
end

gradient(input_indices::AbstractArray) = fullderivative(Val(1), input_indices)
gradient(dim::Integer) = fullderivative(Val(1), dim)

hessian(input_indices::AbstractArray) = fullderivative(Val(2), input_indices)
hessian(dim::Integer) = fullderivative(Val(2), dim)

# idea: lazy mappings can be undone (extract original range -> towards a specialization speedup of broadcasting over multiple derivatives using backwardsdiff)
const MappedPartialVec{T} = ReadonlyMappedArray{Partial{1,Tuple{Int}},1,T,typeof(partial)}
function extract_range(p_map::MappedPartialVec{T}) where {T<:AbstractUnitRange{Int}}
    return p_map.data::T
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
