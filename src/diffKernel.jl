import ForwardDiff as FD
import LinearAlgebra as LA
using KernelFunctions: SimpleKernel, Kernel

const DiffPt = Tuple{Partial,Vararg} # allow for one dimensional and MultiOutput kernels
"""
	diffKernelCall(k::T, (px,x)::DiffPt, (py, y)::DiffPt) where {Dim, T<:Kernel}

specialization for DiffPt. Unboxes the partial instructions from DiffPt and
applies them to k, evaluates them at the positions of DiffPt
"""
function diffKernelCall(k::T, (px, x)::Tuple{Partial,Pos1}, (py, y)::Tuple{Partial,Pos2}) where {T<:Kernel,Pos1,Pos2}
    # need Pos1 and Pos2 because k(1,1.) is allowed (combination of Int and Float) mabye there is a better solution resulting in more type safety?
    return apply_partial(k, px.indices, py.indices)(x, y)
end
"""
Multi Kernel Version (do not try to take the derivative with regard to out indices)
"""
function diffKernelCall(
    k::T,
    (px, x, x_out)::Tuple{Partial,Pos1,Idx1},
    (py, y, y_out)::Tuple{Partial,Pos2,Idx2}
) where {T<:MOKernel,Pos1,Idx1,Pos2,Idx2}
    return apply_partial((x, y) -> k((x, x_out), (y, y_out)), px.indices, py.indices)(x, y)
end

"""
    EnableDiff

A thin wrapper around Kernels enabling the machinery which allows you to
input (∂ᵢ, x), (∂ⱼ, y) where ∂ᵢ, ∂ⱼ are of `Partial` type (see [partial](@ref)) in order
to calculate
``
    k((∂ᵢ, x), (∂ⱼ, y)) = \\text{Cov}(\\partial_i Z(x), \\partial_j Z(y))
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
(k::EnableDiff)(x::DiffPt, y) = diffKernelCall(k.kernel, x, (partial(), y))
(k::EnableDiff)(x, y::DiffPt) = diffKernelCall(k.kernel, (partial(), x), y)
(k::EnableDiff)(x, y) = k.kernel(x, y) # Fall through case 

