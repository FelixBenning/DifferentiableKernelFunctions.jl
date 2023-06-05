module DifferentiableKernelFunctions

using Reexport

@reexport using KernelFunctions

export partial, EnableDiff

include("multiOutput.jl")
include("partial.jl")
include("diffKernel.jl")

end
