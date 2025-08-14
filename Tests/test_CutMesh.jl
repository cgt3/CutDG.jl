using SafeTestsets

include("../Structures/CutMesh.jl")

@safetestset "CutMesh.jl" begin 

@testset "    1D MeshIndex" begin
    index = MeshIndex(1, 2, 3, 4)
    
    @test index.level == 1
    @test index.patch == 2
    @test index.element == 3
    @test index.cut_element == 4
end

end # safetestset