using SafeTestsets

# @testset "    " begin
#     @text
#     @test_throws
# end

@safetestset "CutMesh.jl" begin 


@safetestset "Bounds" begin 
    include("../Structures/CutMesh.jl")
    using .CutMesh

    @testset "LB/UB types must match" begin
        @test_throws "" Bounds(1, 1.0)
    end

    @testset "Length of arrays must match" begin
        @test_throws "" Bounds([1.0], [0.0 1.0])
    end
end # safetestset CartesianDoF


@safetestset "ElementIndex" begin 
    include("../Structures/CutMesh.jl")
    using .CutMesh

    @testset "Integer/integer array indices only" begin
        @test_throws "" ElementIndex(1.0, 1, 1, 1)
        @test_throws "" ElementIndex(1, [1.0, 1.0], 1, 1)
        @test_throws "" ElementIndex(1, 1, 1.0, 1)
        @test_throws "" ElementIndex(1, 1, 1, 1.0)
    end

    @testset "1D: Integer element index" begin
        index = ElementIndex(1, 2, 3, 4)
        
        @test index.level == 1
        @test index.patch == 2
        @test index.element == 3
        @test index.cut_element == 4
    end

    @testset "1D: Cannot use array for 1D index" begin
        @test_throws "" ElementIndex(1, 2, [3, 4], 5, dim=1)
    end

    @testset "nD: element index = nD array" begin
        index = ElementIndex(1, 2, [3,4], 5)

        @test isa(index, ElementIndex{2})
        @test index.level == 1
        @test index.patch == 2
        @test index.element == [3, 4]
        @test index.cut_element == 5
    end


    @testset "nD: element index = integer" begin
        index = ElementIndex(1, 2, 3, 4, dim=2)

        @test isa(index, ElementIndex{2})
        @test index.level == 1
        @test index.patch == 2
        @test index.element == 3
        @test index.cut_element == 4
    end

    @testset "Cartesian index" begin
        index_1D = ElementIndex(1, 2, 3)
        index_nD = ElementIndex(1, 2, [3, 3])

        @test index_1D.cut_element == undef
        @test index_nD.cut_element == undef
    end
end # safetestset ElementIndex


@safetestset "CartesianDoF" begin 
    include("../Structures/CutMesh.jl")
    using .CutMesh

    @testset "Coarest level can have only one patch" begin
        num_levels = 3
        n_by_level = [4; 8; 16]
        patches_by_level = [[Bounds(1,1) Bounds(3,4)], [Bounds(1,2)], [Bounds(1,2)] ]

        @test_throws "" CartesianDoF(num_levels, n_by_level, patches_by_level)
    end

    @testset "Must fully allocate the coarsest level" begin
        num_levels = 3
        n_by_level = [4; 8; 16]
        patches_by_level = [[Bounds(1,2)], [Bounds(1,2)], [Bounds(1,2)] ]

        @test_throws "" CartesianDoF(num_levels, n_by_level, patches_by_level)
    end

    @testset "Grids must conform to coarser grids" begin
        num_levels = 3
        n_by_level = [4; 6; 12]
        patches_by_level = [[Bounds(1,4)], [Bounds(1,2)], [Bounds(1,2)] ]

        @test_throws "" CartesianDoF(num_levels, n_by_level, patches_by_level)
    end

    @testset "1D: Correct level/patch sizes and types" begin
        num_levels = 3
        n_by_level = [4; 8; 16]
        patches_by_level = [[Bounds(1,4)], [Bounds(1,3) Bounds(7,8)], [Bounds(2,2)] ]

        dof = CartesianDoF(num_levels, n_by_level, patches_by_level, elem_type=Int64)

        # Correct number of levels
        @test length(dof.data) == 3

        # Patches have the correct element type
        @test eltype(dof.data[1][1]) == Int64
        @test eltype(dof.data[2][1]) == Int64
        @test eltype(dof.data[2][2]) == Int64
        @test eltype(dof.data[3][1]) == Int64

        # Each level has the correct number of patches
        @test length(dof.data[1]) == 1
        @test length(dof.data[2]) == 2
        @test length(dof.data[3]) == 1

        # Each patch has the correct # of elements
        @test length(dof.data[1][1]) == 4
        @test length(dof.data[2][1]) == 3
        @test length(dof.data[2][2]) == 2
        @test length(dof.data[3][1]) == 1
    end

    @testset "getindex: ElementIndex" begin
        num_levels = 3
        n_by_level = [4; 8; 16]
        patches_by_level = [[Bounds(1,4)], [Bounds(1,3) Bounds(7,8)], [Bounds(2,2)] ]

        dof = CartesianDoF(num_levels, n_by_level, patches_by_level, elem_type=Int64)

        # Test a full index
        index = ElementIndex(2, 2, 1)
        dof.data[index.level][index.patch][index.element] = 2
        @test getindex(dof, index) == 2
        @test dof[index] == 2
    end

        @testset "getindex: slurping" begin
        num_levels = 3
        n_by_level = [4; 8; 16]
        patches_by_level = [[Bounds(1,4)], [Bounds(1,3) Bounds(7,8)], [Bounds(2,2)] ]

        dof = CartesianDoF(num_levels, n_by_level, patches_by_level, elem_type=Int64)

        index = ElementIndex(2, 2, 1)
        dof.data[index.level][index.patch][index.element] = 2

        # Test an index with incorrect length
        @test_throws "" getindex(dof, 1)
        @test_throws "" dof[1] == 2

        # Test a valid index
        @test getindex(dof, 2, 2, 1) == 2
        @test dof[2,2,1] == 2
    end


    @testset "setindex!: ElementIndex" begin
        num_levels = 3
        n_by_level = [4; 8; 16]
        patches_by_level = [[Bounds(1,4)], [Bounds(1,3) Bounds(7,8)], [Bounds(2,2)] ]

        dof = CartesianDoF(num_levels, n_by_level, patches_by_level, elem_type=Int64)

        index = ElementIndex(2, 2, 1)

        setindex!(dof, 4, index)
        @test dof.data[index.level][index.patch][index.element] == 4

        dof[index] = 5
        @test dof.data[index.level][index.patch][index.element] == 5
    end

    @testset "setindex!: slurping" begin
        num_levels = 3
        n_by_level = [4; 8; 16]
        patches_by_level = [[Bounds(1,4)], [Bounds(1,3) Bounds(7,8)], [Bounds(2,2)] ]

        dof = CartesianDoF(num_levels, n_by_level, patches_by_level, elem_type=Int64)

        index = ElementIndex(2, 2, 1)

        setindex!(dof, 5, 2, 2, 1)
        @test dof.data[index.level][index.patch][index.element] == 5

        dof[2,2,1] = 4
        @test dof.data[index.level][index.patch][index.element] == 4
    end
    
end # safetestset CartesianDoF

end
