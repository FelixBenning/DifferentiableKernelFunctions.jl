import ForwardDiff as FD
import LinearAlgebra as LA
using KernelFunctions: SimpleKernel, Kernel

""" 
	DiffPt(x; partial=())

For a covariance kernel k of GP Z, i.e.
```julia
	k(x,y) # = Cov(Z(x), Z(y)),
```
a DiffPt allows the differentiation of Z, i.e.	
```julia
	k(DiffPt(x, partial=1), y) # = Cov(∂₁Z(x), Z(y))
```
for higher order derivatives partial can be any iterable, i.e.
```julia
	k(DiffPt(x, partial=(1,2)), y) # = Cov(∂₁∂₂Z(x), Z(y))
```
"""

IndexType = Union{Int,Base.AbstractCartesianIndex}

struct DiffPt{Order,KeyT<:IndexType,T}
    pos::T # the actual position
    partials::NTuple{Order,KeyT}
end

DiffPt(x::T) where {T<:AbstractArray} = DiffPt{0,keytype(T),T}(x, ()::NTuple{0,keytype(T)})
DiffPt(x::T) where {T<:Number} = DiffPt{0,Int,T}(x, ()::NTuple{0,Int})
DiffPt(x::T, partial::IndexType) where {T} = DiffPt{1,IndexType,T}(x, (partial,))
function DiffPt(x::T, partials::NTuple{Order,KeyT}) where {T,Order,KeyT}
    return DiffPt{Order,KeyT,T}(x, partials)
end


"""
	_evaluate(k::T, x::DiffPt{Dim}, y::DiffPt{Dim}) where {Dim, T<:Kernel}

implements `(k::T)(x::DiffPt{Dim}, y::DiffPt{Dim})` for all kernel types. But since
generics are not allowed in the syntax above by the dispatch system, this
redirection over `_evaluate` is necessary

unboxes the partial instructions from DiffPt and applies them to k,
evaluates them at the positions of DiffPt
"""
function _evaluate(k::T, x::DiffPt, y::DiffPt) where {T<:Kernel}
    return partial(k, x.partials, y.partials)(x.pos, y.pos)
end

#=
This is a hack to work around the fact that the `where {T<:Kernel}` clause is
not allowed for the `(::T)(x,y)` syntax. If we were to only implement
```julia
	(::Kernel)(::DiffPt,::DiffPt)
```
then julia would not know whether to use
`(::SpecialKernel)(x,y)` or `(::Kernel)(x::DiffPt, y::DiffPt)`
```
=#
for T in [SimpleKernel, Kernel] #subtypes(Kernel)
    (k::T)(x::DiffPt, y::DiffPt) = _evaluate(k, x, y)
    (k::T)(x::DiffPt, y) = _evaluate(k, x, DiffPt(y))
    (k::T)(x, y::DiffPt) = _evaluate(k, DiffPt(x), y)
end
