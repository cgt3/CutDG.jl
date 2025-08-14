struct MeshIndex{dim}
    level::Integer
    patch::Integer
    element::Union{AbstractArray, Integer}
    cut_element::Integer

    # TODO: If length(I_element) == 1, should it still be stored as an array?
    MeshIndex(i_level::Integer, i_patch::Integer, I_element::AbstractArray, i_cut::Integer) = 
        new{length(I_element)}(i_level, i_patch, I_element, i_cut)
    MeshIndex(i_level::Integer, i_patch::Integer, i_element::Integer, i_cut::Integer; dim=1) = 
        new{dim}(i_level, i_patch, i_element, i_cut)
end

# Constructors for partial indices:
MeshIndex(i_level, i_patch, I_element) = MeshIndex(i_level, i_patch, I_element, undef)
MeshIndex(i_level, i_patch) = MeshIndex(i_level, i_patch, undef, undef)
MeshIndex(i_level) = MeshIndex(i_level, undef, undef, undef)


abstract type AbstractDoF end
abstract type AbstractCartesianDoF <: AbstractDoF end # These can only have 1 set of DoF per background element
abstract type AbstractCutDoF <: AbstractDoF end # These can contain multiple sets of DoF per background element

struct CartesianDoF{dim} <: AbstractCartesianDoF 
    data

    # num_levels = the number of levels
    # n_by_level = the number of elements in each direction per level: n_by_level[i_level, dir]
    # patches_by_level = 
    function CartesianDoF(num_levels, n_by_level, patches_by_level; eltype=Float64)
        # Error checking
        n_prev = n_by_level[1,:]
        for i_level in 2:size(n_by_level,1)
            # Check that the levels are conforming
            if any(n_by_level[i_level, :] .% n_prev != 0)
                throw(ErrorException("CartesianDoF: levels must be conforming: n[level=$i_level] = $(n_by_level[i_level,:]), n[level=$(i_level-1)] = $n_prev"))
            end

            # Check that the patch indices are within bounds
            if any(patches_by_level[i_level].lb .< 1) || any(patches_by_level[i_level].ub .> n_by_level[i_level,:])
                throw(ErrorException("CartesianDoF: patch indices (I=$(patches_by_level[i_level])) are out of bounds (n_xyz=$(n_by_level[i_level,:]))"))
            end
        end

        # Allocate memory
        dim = size(n_by_level,2)
        data = Vector{Matrix{eltype}}[] # Empty Vector{Vector{Matrix{eltype}}}
        for i_level in 1:num_levels
            patch_mem_level_i = Matrix{eltype}[]
            for i_patch in eachindex(patches_by_level[i_level])
                patch_size = patches_by_level.ub .- patches_by_level.lb .+ 1
                push!(patch_mem_level_i, zeros(patch_size...))
            end
            
            push!(data, patch_mem_level_i)
        end

        return new{dim}(data)
    end
end

# struct SparseCartesianDoF <: AbstractCartesianDoF
# end

# struct CutDoF <: AbstractCutDoF
# end

# struct SparseCutDoF <: AbstractCutDoF
# end

function Base.:getindex(dof::AbstractCartesianDoF, index::MeshIndex)
    return dof.data[index.level][index.patch][index.cart_element]
end

function Base.:getindex(dof::AbstractCutDoF)

end