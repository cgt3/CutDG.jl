module CutMesh

# Types
export ElementIndex, Bounds, CartesianDoF


struct Bounds{T}
    lb::T
    ub::T

    function Bounds(lb::T, ub::T) where T
        if isa(lb, AbstractArray) && length(lb) != length(ub)
            throw(ErrorException("Bounds: Lower bound array and upper bounds array have different lengths"))
        end
        return new{T}(lb, ub)
    end
end


struct ElementIndex{dim}
    level::Integer
    patch::Integer
    element::Union{AbstractArray, Integer}
    cut_element::Union{Integer, UndefInitializer}

    # TODO: If length(I_element) == 1, should it still be stored as an array?
    ElementIndex(i_level::Integer, i_patch::Integer, I_element::AbstractArray, i_cut::Union{Integer, UndefInitializer}) = 
        new{length(I_element)}(i_level, i_patch, I_element, i_cut)

    ElementIndex(i_level::Integer, i_patch::Integer, 
        i_element::Integer, 
        i_cut::Union{Integer, UndefInitializer}; 
        dim::Integer) =  new{dim}(i_level, i_patch, i_element, i_cut)
end

ElementIndex(i_level::Integer, i_patch::Integer, i_element::Integer, i_cut::Integer) = ElementIndex(i_level, i_patch, i_element, i_cut, dim=1)
ElementIndex(i_level::Integer, i_patch::Integer, I_element::AbstractArray) = ElementIndex(i_level, i_patch, I_element, undef)
ElementIndex(i_level::Integer, i_patch::Integer, I_element::Integer) = ElementIndex(i_level, i_patch, I_element, undef, dim=length(I_element))


abstract type AbstractDoF end
abstract type AbstractCartesianDoF <: AbstractDoF end # These can only have 1 set of DoF per background element
abstract type AbstractCutDoF <: AbstractDoF end # These can contain multiple sets of DoF per background element

struct CartesianDoF{dim, T_elem} <: AbstractCartesianDoF 
    data
    nxyz

    # num_levels = the number of levels
    # nxyz = the number of elements in each direction per level: nxyz[i_level, dir]
    # patches_by_level = 
    function CartesianDoF(num_levels, nxyz, patches_by_level; elem_type=Float64)
        # Check that the coarsest mesh is fully allocated
        if length(patches_by_level[1]) != 1
            throw(ErrorException("CartesianDoF: Coarsest level can only have one patch."))
        elseif any(patches_by_level[1][1].lb .!= 1) || any(patches_by_level[1][1].ub .!= nxyz[1])
            throw(ErrorException("CartesianDoF: Base patch does not cover the whole domain"))
        end

        n_prev = nxyz[1,:]
        for i_level in 2:size(nxyz,1)
            # TODO: Check that grids are ordered coarsest to finest

            # Check that the levels use conforming grids
            if any(nxyz[i_level, :] .% n_prev .!= 0)
                throw(ErrorException("CartesianDoF: levels must be conforming: n[level=$i_level] = $(nxyz[i_level,:]), n[level=$(i_level-1)] = $n_prev"))
            end

            # Check that the patch indices are within bounds
            for i_patch in eachindex(patches_by_level[i_level])
                if any(patches_by_level[i_level][i_patch].lb .< 1) || any(patches_by_level[i_level][i_patch].ub .> nxyz[i_level,:])
                    throw(ErrorException("CartesianDoF: patch indices (I=$(patches_by_level[i_level])) are out of bounds (n_xyz=$(nxyz[i_level,:]))"))
                end
            end

            # TODO: Check that patches do not overlap

            # TODO: Check that fine patches are confined to regions covered by coarser patches
        end

        # Allocate memory
        dim = size(nxyz,2)
        data = Vector{Matrix{elem_type}}[] # Empty Vector{Vector{Matrix{elem_type}}}
        for i_level in 1:num_levels
            patch_mem_level_i = Matrix{elem_type}[]
            for i_patch in eachindex(patches_by_level[i_level])
                patch_size = patches_by_level[i_level][i_patch].ub .- patches_by_level[i_level][i_patch].lb .+ 1
                if length(patch_size) == 1
                    push!(patch_mem_level_i, zeros(elem_type, patch_size, 1))
                else
                    push!(patch_mem_level_i, zeros(elem_type, patch_size...))
                end
            end
            
            push!(data, patch_mem_level_i)
        end

        return new{dim, elem_type}(data, nxyz)
    end
end

# struct SparseCartesianDoF <: AbstractCartesianDoF
# end

# struct CutDoF <: AbstractCutDoF
# end

# struct SparseCutDoF <: AbstractCutDoF
# end

@inline function Base.:getindex(dof::AbstractCartesianDoF, index::ElementIndex)
    return dof.data[index.level][index.patch][index.element]
end

@inline function Base.:getindex(dof::AbstractCartesianDoF, ind...)
    if length(ind) < 3
        throw(ErrorException("getindex(CartesianDoF): index must have at least length 3"))
    end

    return dof.data[ ind[1] ][ ind[2] ][ ind[3] ]
end


@inline function Base.:setindex!(dof::AbstractCartesianDoF, x, index::ElementIndex)
    dof.data[index.level][index.patch][index.element] = x
    return x
end

@inline function Base.:setindex!(dof::AbstractCartesianDoF, x, ind...)
    if length(ind) < 3
        throw(ErrorException("setindex!(CartesianDoF): index must have at least length 3"))
    end
    
    dof.data[ ind[1] ][ ind[2] ][ ind[3] ] = x
    return x
end

end # CutMesh module