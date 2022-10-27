using Aqua
using Graphs
using JuliaFormatter
using MinimumWeightTwoStageSpanningTree
using Test

format(MinimumWeightTwoStageSpanningTree; verbose=true)

@testset verbose = true "MinimumWeightTwoStageSpanningTree.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(
            MinimumWeightTwoStageSpanningTree;
            deps_compat=false,
            project_extras=true,
            ambiguities=false,
        )
    end
    @testset "Test Cut Generation, Benders and Lagrangian relaxation on a TwoStageSpanningTree instance with a single scenario" begin
        tol = 0.00001
        instance = MinimumWeightTwoStageSpanningTree.TwoStageSpanningTreeInstance(
            grid([3, 3]); nb_scenarios=1, first_max=10, second_max=10
        )
        kruskal_value, _, _ = MinimumWeightTwoStageSpanningTree.kruskal_on_first_scenario_instance(
            instance
        )
        benders_value, forest, theta = benders_decomposition(instance; silent=true)
        @test abs(kruskal_value - benders_value) < tol

        cl_val, cl_theta = column_generation(instance)

        lb, ub, _, _ = lagrangian_relaxation(instance; nb_epochs=20000)
        @test lb <= ub
        @test lb <= kruskal_value
        @test ub >= kruskal_value
        @test abs(lb - kruskal_value) <= 0.1
        @test abs(
            MinimumWeightTwoStageSpanningTree.lagrangian_dual(
                -instance.nb_scenarios * cl_theta; inst=instance
            ) - cl_val,
        ) <= 0.00001

        cut_value, forest = cut_generation(instance; silent=true)
        @test abs(benders_value - cut_value) < 0.0001
    end
end
