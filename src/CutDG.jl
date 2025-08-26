module CutDG

# Types
export ElementIndex, Bounds, CartesianDoF

# Functions
export getlevel, getpatch

export ref2phys, phys2ref, get_element_bounds, get_operator_scaling, get_elem_mass, 
export get_superbounds, get_mixed_mass_matrix, get_new_boundary, merge_elements

abstract type AbstractRegionBoundary end

# TODO: Should Bounds also track their active/inactive dimensions?
struct Bounds{dim, T} <: AbstractRegionBoundary
    lb::T
    ub::T
    isactive::AbstractArray

    # TODO: for integer Bounds isactive does not make sense
    function Bounds(lb::T, ub::T; active_tol=1e-14) where T
        if isa(lb, AbstractArray) && length(lb) != length(ub)
            throw(ErrorException("Bounds: Lower bound array and upper bounds array have different lengths"))
        end

        dim = length(lb)
        if dim > 1
            isactive = zeros(Bool, length(lb))
            for i in eachindex(lb)
                if lb[i] > ub[i]
                    throw(ErrorException("Bounds: lower bounds cannot be greater than upper bounds."))
                elseif abs(lb[i] - ub[i]) < 1e-14
                    isactive[i] = false
                else
                    isactive[i] = true
                end
            end
        elseif lb > ub
            throw(ErrorException("Bounds: lower bounds cannot be greater than upper bounds."))
        else
            isactive = true
        end

        return new{dim, T}(lb, ub)
    end
end

# TODO: make a const Bounds object of the reference domain?

struct CutElement{dim}
    boundary::AbstractRegionBoundary
    CutElement(bounds::Bounds) = new{1}(bounds)
end

# TODO: add a parameter that denotes whether an instances is a cut or Cartesian index?
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
        data = Vector{Matrix{elem_type}}[] # Empty Vector{Vector{Matrix{elem_type}}}: data[level][patch][i,j]
        for i_level in 1:num_levels
            patch_mem_level_i = Matrix{elem_type}[]
            for i_patch in eachindex(patches_by_level[i_level])
                patch_size = patches_by_level[i_level][i_patch].ub .- patches_by_level[i_level][i_patch].lb .+ 1
                if length(patch_size) == 1
                    push!(patch_mem_level_i, zeros(elem_type, patch_size, 1)) # BUG: elem_type can be a vector
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

struct CutDoF{dim} <: AbstractCutDoF
    data 
    nxyz

    function CutDoF(num_levels, nxyz, patches_by_level, cuts::Vector{<:Real}; elem_type=Float64)
        # Check that the coarsest mesh is fully allocated
        if length(patches_by_level[1]) != 1
            throw(ErrorException("CutDoF: Coarsest level can only have one patch."))
        elseif any(patches_by_level[1][1].lb .!= 1) || any(patches_by_level[1][1].ub .!= nxyz[1])
            throw(ErrorException("CutDoF: Base patch does not cover the whole domain"))
        end

        n_prev = nxyz[1,:]
        for i_level in 2:size(nxyz,1)
            # TODO: Check that grids are ordered coarsest to finest

            # Check that the levels use conforming grids
            if any(nxyz[i_level, :] .% n_prev .!= 0)
                throw(ErrorException("CutDoF: levels must be conforming: n[level=$i_level] = $(nxyz[i_level,:]), n[level=$(i_level-1)] = $n_prev"))
            end

            # Check that the patch indices are within bounds
            for i_patch in eachindex(patches_by_level[i_level])
                if any(patches_by_level[i_level][i_patch].lb .< 1) || any(patches_by_level[i_level][i_patch].ub .> nxyz[i_level,:])
                    throw(ErrorException("CutDoF: patch indices (I=$(patches_by_level[i_level])) are out of bounds (n_xyz=$(nxyz[i_level,:]))"))
                end
            end

            # TODO: Check that patches do not overlap

            # TODO: Check that fine patches are confined to regions covered by coarser patches? 
            #  - Only if neighbor refinement level restrictions are imposed
        end

        # Setup the nested data structure
        dim = size(nxyz,2)
        data = Vector{Matrix{Vector{elem_type}}}[] # Empty Vector{Vector{Matrix{Vector{elem_type}}}}: data[level][patch][i,j][cut]
        for i_level in 1:num_levels
            patch_mem_level_i = Matrix{Vector{elem_type}}[]
            for i_patch in eachindex(patches_by_level[i_level])
                patch_size = patches_by_level[i_level][i_patch].ub .- patches_by_level[i_level][i_patch].lb .+ 1
                if length(patch_size) == 1
                    push!(patch_mem_level_i, Matrix{Vector{elem_type}}(undef, patch_size, 1))
                else
                    push!(patch_mem_level_i, Matrix{Vector{elem_type}}(undef, patch_size...))
                end
            end
            
            push!(data, patch_mem_level_i)
        end

        # Rule of thumb: cuts cut just the leaf level
        # Allocate memory for cut elements on every patch/level
        for cut in cuts 
            for i_level in reverse(eachindex(data)) # Start at the finest level
                I_cut = get_cartesian_indices(cut.val, nxyz[i_level], domain)

                if length(I_cut) == 1 # The cut does not conform to the background mesh
                    for i_patch in eachindex(patches_by_level[i_level])
                        # TODO: Replace this linear search with the BVH
                        # Check if the element containing this cut is in this patch
                        if patch.lb <= I_cut <= patch.ub
                            if isempty(data[i_level][i_patch][I_cut])
                                push!(data[i_level][i_patch][I_cut], zeros(elem_type)) # BUG: elem_type can be a vector
                            end
                            data[i_level][i_patch][I_cut] = zeros(elem_type, )
                        end
                    end
                end
            end
        end

        return new{dim, elem_type}(data, nxyz)
    end
end

# struct SparseCutDoF <: AbstractCutDoF
# end

# Notes:
# - p   : is to be inferred from the DOF or stored in ref_oeprators to allow 
#         p-adaptivity
# - nxyz: is to be inferred from the DOF or stored in the ref_operators
# TODO: add neighbor mapping
struct CutDGSolution{dim, T_cartElem, T_cutElem}
    cart_dof::CartesianDoF{dim, T_cartElem}
    cut_dof::CutDoF{dim, T_cutElem}
    cut_elements::Vector{CutElement{dim}}
    cut_indices::Vector{ElementIndices{dim}}
    ref_operators
    # TODO: neighbor mapping
end

@inline function getlevel(dof::AbstractDoF, i_level::Integer)
    return dof.data[i_level]
end

@inline function getpatch(dof::AbstractDoF, ind...)
    if length(ind) < 2
        throw(ErrorException("CutMesh:getpatch: ind must be length 2."))
    end

    return dof.data[ ind[1] ][ ind[2] ]
end

@inline function get_cartesian_elem(dof::AbstractCutDoF, ind...)
    if length(ind) < 3
        throw(ErrorException("CutMesh:get_cartesian_elem: ind must be length 3."))
    end

    return dof.data[ ind[1] ][ ind[2] ][ ind[3] ]
end




# TODO: add functionality to add levels/patches

# The rule of thumb:to avoid confusion, direct indexing is for indiviual elements only
# partial indices cannot 

@inline function Base.:getindex(dof::AbstractCartesianDoF, index::ElementIndex)
    return dof.data[index.level][index.patch][index.element]
end

@inline function Base.:getindex(dof::AbstractCutDoF, index::ElementIndex)
    return dof.data[index.level][index.patch][index.element][index.cut_element]
end

@inline function Base.:getindex(dof::AbstractCartesianDoF, ind...)
    if length(ind) != 3
        throw(ErrorException("getindex(CartesianDoF): ind must be length 3."))
    end

    return dof.data[ ind[1] ][ ind[2] ][ ind[3] ]
end

@inline function Base.:getindex(dof::AbstractCutnDoF, ind...)
    if length(ind) != 4
        throw(ErrorException("getindex(CutDoF): ind must be length 4."))
    end

    return dof.data[ ind[1] ][ ind[2] ][ ind[3] ][ ind[4] ]
end

@inline function Base.:setindex!(dof::AbstractCartesianDoF, x, index::ElementIndex)
    dof.data[index.level][index.patch][index.element] = x
    return x
end

@inline function Base.:setindex!(dof::AbstractCutDoF, x, index::ElementIndex)
    dof.data[index.level][index.patch][index.element][index.cut_element] = x
    return x
end

@inline function Base.:setindex!(dof::AbstractCartesianDoF, x, ind...)
    if length(ind) != 3
        throw(ErrorException("setindex!(CartesianDoF): ind must be length 3."))
    end
    
    dof.data[ ind[1] ][ ind[2] ][ ind[3] ] = x
    return x
end

@inline function Base.:setindex!(dof::AbstractCutDoF, x, ind...)
    if length(ind) != 4
        throw(ErrorException("setindex!(CutDoF): ind must be length 4."))
    end
    
    dof.data[ ind[1] ][ ind[2] ][ ind[3] ][ ind[4] ] = x
    return x
end

# Functions for working with DG solutions ==================================================================================

@inline function ref2phys(r, bounds::Bounds, ; ref_bounds=Bounds(-1.0, 1.0))
    return (bounds.ub - bounds.lb) * (r .- ref_bounds.lb) / (ref_bounds.ub - ref_bounds.lb) .+ bounds.lb
end

@inline function phys2ref(x, bounds::Bounds; ref_bounds=Bounds(-1.0, 1.0))
    return (ref_bounds.ub - ref_bounds.lb) * (x .- bounds.lb) / (bounds.ub - bounds.lb).+ ref_bounds.lb
end


function get_element_bounds(domain_lb, dx, k)
    return Bounds(domain_lb + dx*(k-1), domain_lb + dx*k)
end

function get_operator_scaling(bounds::Bounds; ref_domain=Bounds(-1.0, 1.0))
    return (bounds.ub - bounds.lb) / (ref_domain.ub - ref_domain.lb)
end

function get_operator_scaling(dx::Real; ref_bounds=Bounds(-1.0, 1.0))
    return dx / (ref_bounds.ub - ref_bounds.lb)
end

function get_elem_mass(bounds, U, operators; ref_domain=(-1.0, 1.0))
    U_const_one = operators.Pq * ones(length(U))
    return  get_operator_scaling(bounds, ref_domain=ref_domain) * U_const_one' * operators.M * U
end


# A mixed-mass matrix calculates inner products:
# u_super' * M_mixed * u_sub = inner product over the subdomain
# M_mixed*u_sub = M_super*u_super -> u_sub = M_mixed^(-1) * M_super * u_super
function get_mixed_mass_matrix(subdomain, superdomain, operators; ref_domain = Bounds(-1.0, 1.0))
    # Precompute the lengths of each interval
    L_ref = ref_domain.ub - ref_domain.lb;
    L_super = superdomain.ub - superdomain.lb;
    L_sub = subdomain.ub - subdomain.lb;

    # Map the subdomain's nodes to the reference domain of the superdomain
    r_sub_lb = L_ref * (subdomain.lb - superdomain.lb) / L_super + ref_domain.lb;
    r_sub = L_sub / L_super * (operators.r .- ref_domain.lb)  .+ r_sub_lb;

    # Construct the Vandermonde matrix of the superbasis evaluated at the subdomain's 
    # nodes in the superdomain
    V_sub = vandermonde(Line(), rd.N, r_sub) / operators.VDM;

    return L_sub / L_ref * V_sub' * operators.M # should be L_sub / L_super?
end

function get_superbounds(bounds1, bounds2)
    return Bounds(min(bounds1.lb, bounds2.lb), max(bounds1.ub, bounds2.ub))
end


function merge_elements(U1, bounds1, U2, bounds2; ref_domain=Bounds(-1.0, 1.0))
    bounds_merged = get_superbounds(bounds1, bounds2)

    # Calculate the moments from each solution on the superinterval
    m1 = get_mixed_mass_matrix(bounds1, bounds_merged, operators; ref_domain=ref_domain) * U1
    m2 = get_mixed_mass_matrix(bounds2, bounds_merged, operators; ref_domain=ref_domain) * U2
    m_merged = m1 + m2

    U_merged = operators.M \ m_merged / get_operator_scaling(bounds_merged; ref_domain=ref_domain)

    return U_merged, bounds_merged
end


function get_new_boundary(domain_orig, u_orig, uf_ext_L, operators; dx_tol=1e-6, penalty_weights=[0.5, 0.5], ref_domain=Bounds(-1.0, 1.0))
    # Find the moments of u_orig
    moments_orig = get_operator_scaling(domain_orig) * operators.M * u_orig 
    uf_true = [uf_ext_L, 0.0]

    # For checking positivity:
    dr_dense = 1e-4
    r_dense = ref_domain.lb:dr_dense:ref_domain.ub
    V_pos = vandermonde(Line(), length(operators.r)-1, r_dense) / operators.VDM

    # Reduce the interval size until the new solution is positive
    penalty_min = 2 * uf_ext_L
    u_opt = similar(u_orig)

    # Brute force search to tolerance dx
    dx = 1e-3
    x_ub = domain_orig.ub
    x_lb = domain_orig.lb + dx
    x_cut_opt = x_lb
    while dx >= dx_tol
        for x_cut in x_lb:dx:x_ub
            domain_new = Bounds(domain_orig.lb, x_cut)
            M_mixed = get_mixed_mass_matrix(domain_new, domain_orig, operators, ref_domain=ref_domain)
            u_new = M_mixed \ moments_orig
            is_pos = all(V_pos * u_new .>= 0)
            if is_pos
                penalty = penalty_weights' * abs.(operators.Vf*u_new - uf_true)
                if penalty < penalty_min
                    penalty_min = penalty
                    u_opt .= u_new
                    x_cut_opt = x_cut
                end
            end
        end
        x_ub = x_cut_opt + dx
        x_lb = x_cut_opt - dx
        dx = dx / 10
    end

    # Return the minimizer
    domain_new = Bounds(domain_orig.lb, x_cut_opt);
    M_mixed = get_mixed_mass_matrix(domain_new, domain_orig, operators, ref_domain=ref_domain);
    u_new = M_mixed \ moments_orig;
    penalty = penalty_weights' * abs.(operators.Vf*u_new - uf_true)

    return u_new, x_cut_opt, penalty
end


end # CutMesh module