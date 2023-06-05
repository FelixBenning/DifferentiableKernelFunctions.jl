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
    EnableDiffWrap

A thin wrapper around Kernels enabling the machinery which allows you to
input (x, ∂ᵢ), (y, ∂ⱼ) where ∂ᵢ, ∂ⱼ are of `Partial` type (see [partial](@ref)) in order
to calculate
``
    k((x, ∂ᵢ), (y,∂ⱼ)) = \\text{Cov}(\\partial_i Z(x), \\partial_j Z(y))
``
for ``Z`` with ``k(x,y) = \\text{Cov}(Z(x), Z(y))``.

While this machinery could in principle be enabled for all `Kernel` by default,
the covariance of derivatives of an isotropic kernel are no longer isotropic.
This forces the use of less specialized methods. So you have to activate it with this Wrapper.

Example:

```jldoctest

```
"""
struct EnableDiffWrap{T<:Kernel} <: Kernel
    kernel::T
end
(k::EnableDiffWrap)(x::DiffPt, y::DiffPt) = diffKernelCall(k.kernel, x, y)
(k::EnableDiffWrap)(x::DiffPt, y) = diffKernelCall(k.kernel, x,(y, partial()))
(k::EnableDiffWrap)(x, y::DiffPt) = diffKernelCall(k.kernel, (x, partial()), y)
(k::EnableDiffWrap)(x, y) = k.kernel(x,y) # Fall through case 

