import ForwardDiff as FD
import LinearAlgebra as LA
using KernelFunctions: SimpleKernel, Kernel

"""
	_evaluate(k::T, x::DiffPt{Dim}, y::DiffPt{Dim}) where {Dim, T<:Kernel}

implements `(k::T)(x::DiffPt{Dim}, y::DiffPt{Dim})` for all kernel types. But since
generics are not allowed in the syntax above by the dispatch system, this
redirection over `_evaluate` is necessary

unboxes the partial instructions from DiffPt and applies them to k,
evaluates them at the positions of DiffPt
"""
function _evaluate(k::T, (x,px)::DiffPt, (y,py)::DiffPt) where {T<:Kernel}
    return apply_partial(k, px.indices, py.indices)(x, y)
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
    (k::T)(x::DiffPt, y) = _evaluate(k, x,(y, partial()))
    (k::T)(x, y::DiffPt) = _evaluate(k, (x, partial()), y)
end
