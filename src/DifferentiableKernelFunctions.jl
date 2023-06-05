module DifferentiableKernelFunctions

using Reexport

@reexport using KernelFunctions

export partial, EnableDiffWrap

include("multiOutput.jl")
include("partial.jl")
include("diffKernel.jl")

end
