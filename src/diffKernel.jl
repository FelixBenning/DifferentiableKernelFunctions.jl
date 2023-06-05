import ForwardDiff as FD
import LinearAlgebra as LA
using KernelFunctions: SimpleKernel, Kernel

"""
	diffKernelCall(k::T, (x,px)::DiffPt, (y,py)::DiffPt) where {Dim, T<:Kernel}

specialization for DiffPt. Unboxes the partial instructions from DiffPt and
applies them to k, evaluates them at the positions of DiffPt
"""
function diffKernelCall(k::T, (x,px)::DiffPt, (y,py)::DiffPt) where {T<:Kernel}
    return apply_partial(k, px.indices, py.indices)(x, y)
end

"""
    EnableDiff

A thin wrapper around Kernels enabling the machinery which allows you to
input (x, ∂ᵢ), (y, ∂ⱼ) where ∂ᵢ, ∂ⱼ are of `Partial` type (see [partial](@ref)) in order
to calculate
``
    k((x, ∂ᵢ), (y,∂ⱼ)) = \\text{Cov}(\\partial_i Z(x), \\partial_j Z(y))
``
for ``Z`` with ``k(x,y) = \\text{Cov}(Z(x), Z(y))``.

!!! warning Only apply this wrapper at the very end. Kerneltransformations
should be applied beforehand.

!!! info While this machinery could in principle be enabled for all `Kernel` by default,
the covariance of derivatives of an isotropic kernel are no longer isotropic.
This forces the use of less specialized methods. So for now you have to opt-in
with this Wrapper.

Example:

```jldoctest
julia> k = EnableDiff(SEKernel());

julia> k((0, partial(1)), 0) # calculate Cov(∂₁Z(0), Z(0))
0.0

julia> k(0,0) # normal input still works
1.0
```
"""
struct EnableDiff{T<:Kernel} <: Kernel
    kernel::T
end
(k::EnableDiff)(x::DiffPt, y::DiffPt) = diffKernelCall(k.kernel, x, y)
(k::EnableDiff)(x::DiffPt, y) = diffKernelCall(k.kernel, x,(y, partial()))
(k::EnableDiff)(x, y::DiffPt) = diffKernelCall(k.kernel, (x, partial()), y)
(k::EnableDiff)(x, y) = k.kernel(x,y) # Fall through case 

