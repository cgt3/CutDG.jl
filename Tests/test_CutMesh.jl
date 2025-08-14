using SafeTestsets


# @testset "    " begin
#     @text
#     @test_throws
# end

@safetestset "MeshIndex" begin 

include("../Structures/CutMesh.jl")

@testset "    Integer/integer array indices only" begin
    @test_throws "" MeshIndex(1.0, 1, 1, 1)
    @test_throws "" MeshIndex(1, [1.0, 1.0], 1, 1)
    @test_throws "" MeshIndex(1, 1, 1.0, 1)
    @test_throws "" MeshIndex(1, 1, 1, 1.0)
end

@testset "    1D: Integer element index" begin
    index = MeshIndex(1, 2, 3, 4)
    
    @test index.level == 1
    @test index.patch == 2
    @test index.element == 3
    @test index.cut_element == 4
end

@testset "    1D: Cannot use array for 1D index" begin
    @test_throws "" MeshIndex(1, 2, [3, 4], 5, dim=1)
end

@testset "    nD: element index = nD array" begin
    index = MeshIndex(1, 2, [3,4], 5)

    @test isa(index, MeshIndex{2})
    @test index.level == 1
    @test index.patch == 2
    @test index.element == [3, 4]
    @test index.cut_element == 5
end


@testset "    nD: element index = integer" begin
    index = MeshIndex(1, 2, 3, 4, dim=2)

    @test isa(index, MeshIndex{2})
    @test index.level == 1
    @test index.patch == 2
    @test index.element == 3
    @test index.cut_element == 4
end

@testset "    Cartesian index" begin
    index_1D = MeshIndex(1, 2, 3)
    index_nD = MeshIndex(1, 2, [3, 3])

    @test index_1D.cut_element == undef
    @test index_nD.cut_element == undef
end

end # safetestset