using KernelFunctions: KernelFunctions as KF, MaternKernel, SEKernel, RationalQuadraticKernel, Matern32Kernel, Matern52Kernel
using DifferentiableKernelFunctions: DifferentiableKernelFunctions as DKF, EnableDiff, partial
using ProductArrays: productArray
using Test

"""
List of Testfiles without extension. `\$(test).jl"` should be a file for every test in AVAILABLE_TESTS
"""
const AVAILABLE_TESTS = [
    "diffKernel",
    "partial"
]

function test_selection()
    group_str = get(ENV, "TESTSET", missing)
    if ismissing(group_str)
        return AVAILABLE_TESTS
    else
        return split(group_str, ",")
    end
end


# To select a specific set the "TESTSET" variable (in particular you can also set it to "folder/subfolder/test" not in AVAILABLE_TESTS)
@testset "DifferentiableKernelFunctions.jl" begin
    for test in test_selection()
        include("$(test).jl")
    end
end
