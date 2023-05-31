using KernelFunctions: KernelFunctions as KF, MaternKernel, SEKernel 
using DifferentiableKernelFunctions: DifferentiableKernelFunctions as DKF, DiffPt, partial
using Test

const AVAILABLE_TESTS = [
    "multiOutput",
    "diffKernel",
]

function test_selection()
    group_str = get(ENV, "GROUP", missing)
    if ismissing(group_str)
        return AVAILABLE_TESTS
    else
        groups = split(group_str, ",")
        surprises = setdiff(groups, AVAILABLE_TESTS)
        isempty(surprises) || throw(ArgumentError("Test groups $surprises not available"))
        return groups 
    end
end


@testset "DifferentiableKernelFunctions.jl" begin
    for test in test_selection()
        include("$(test).jl")
    end
end
