using DifferentiableKernelFunctions
using Documenter

DocMeta.setdocmeta!(DifferentiableKernelFunctions, :DocTestSetup, :(using DifferentiableKernelFunctions); recursive=true)

makedocs(;
    modules=[DifferentiableKernelFunctions],
    authors="Felix Benning <felix.benning@gmail.com> and contributors",
    repo="https://github.com/FelixBenning/DifferentiableKernelFunctions.jl/blob/{commit}{path}#{line}",
    sitename="DifferentiableKernelFunctions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FelixBenning.github.io/DifferentiableKernelFunctions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/FelixBenning/DifferentiableKernelFunctions.jl",
    devbranch="main",
)
