using MinimumWeightTwoStageSpanningTree
using Documenter

DocMeta.setdocmeta!(
    MinimumWeightTwoStageSpanningTree,
    :DocTestSetup,
    :(using MinimumWeightTwoStageSpanningTree);
    recursive=true,
)

makedocs(;
    modules=[MinimumWeightTwoStageSpanningTree],
    authors="Léo Baty",
    repo="https://github.com/BatyLeo/MinimumWeightTwoStageSpanningTree.jl/blob/{commit}{path}#{line}",
    sitename="MinimumWeightTwoStageSpanningTree.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://BatyLeo.github.io/MinimumWeightTwoStageSpanningTree.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Problem statement" => "problem.md",
        "Optimization algorithms" => "optimization.md",
        "inferopt.md",
        "paper.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/BatyLeo/MinimumWeightTwoStageSpanningTree.jl", devbranch="main"
)
