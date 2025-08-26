# TODO: Should Bounds also track their active/inactive dimensions?
struct Bounds{dim, T}
    lb::T
    ub::T
    isactive::Union{AbstractArray, Bool}

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

            return new{dim, T}(lb, ub, isactive)
        elseif lb > ub
            throw(ErrorException("Bounds: lower bounds cannot be greater than upper bounds."))
        else
            isactive = true
            return new{dim, T}(lb, ub, isactive)
        end

    end
end


struct CutDGSolution
    cart_dof
    cut_dof
    cuts
    iscut
    cut_bounds
    domain
    p
    n_elem
end


function plot_basis_vectors(rd; ref_elem=Bounds(-1, 1))
    basis_vec_plot = plot(xlim=(ref_elem.lb, ref_elem.ub), title="Basis Vectors")
    e_i = zeros(p+1)
    for i in 1:length(rd.r)
        e_i[i] = 1
        plot!(basis_vec_plot, rd.rp, rd.Vp * e_i, label="phi_$i")
        e_i[i] = 0
    end
    display(basis_vec_plot)

    return basis_vec_plot
end


function plot_DG_solution(U::CutDGSolution, rd, domain; ylims=(0.0,3.0), field=1)
    field = 1
    IC_plot = plot(xlim=(domain.lb, domain.ub), ylim=ylims, title="IC: p=$p, nx=$nx", legend=false)
    for k in 1:U.n_elem
        if !U.iscut[k] # Cartesian element
            bounds_k = get_element_bounds(domain.lb, dx, k)
            plot!(IC_plot, ref2phys(rd.rp, bounds_k), rd.Vp * getindex.(U.cart_dof[:,k],field), color=:blue)
            vline!(IC_plot, [bounds_k.lb], color=:grey80)
        else # cut element
            for i in eachindex(U.cut_dof[k])
                bounds_k_i = U.cut_bounds[k][i]
                plot!(IC_plot, ref2phys(rd.rp, bounds_k_i), rd.Vp * getindex.(U.cut_dof[k][i],field), color=:blue)
                vline!(IC_plot, [bounds_k_i.lb], color=:grey80)
            end
        end
    end
    # display(IC_plot)
    return IC_plot
end


function plot_DG_solution(U::AbstractArray, rd, domain; ylims=(0.0,3.0), field=1)
    field = 1
    IC_plot = plot(xlim=(domain.lb, domain.ub), ylim=ylims, title="IC: p=$p, nx=$nx", legend=false)
    for k in eachindex(U)
        bounds_k = get_element_bounds(domain.lb, dx, k)
        plot!(IC_plot, ref2phys(rd.rp, bounds_k), rd.Vp * getindex.(U[:,k],field), color=:blue)
        vline!(IC_plot, [bounds_k.lb], color=:grey80)
    end
    # display(IC_plot)
    return IC_plot
end


function rhs!(dUdt, U, t, params)
    # dUdt .= SVector(0.0, 0.0, 0.0)
    vol_terms = 0 .* U[:,1]
    boundary_terms = 0 .* U[:,1]

    # UL = U[:,end] #
    UL = params.BC(U[:,1], params.domain.lb, t)
    for k in axes(U,2)
        bounds_k = get_element_bounds(params.domain.lb, params.dx, k)
        scaling_k = get_operator_scaling(bounds_k)
        xq_k = ref2phys(params.operators.rq, bounds_k)
        
        vol_terms .*= 0;
        for i in axes(params.operators.Q,1)
            for j in 1:i 
                QoF_ij = 2 * params.operators.Q[i,j] * params.f_volume(U[i,k], U[j,k])
                vol_terms[i] += QoF_ij
                vol_terms[j] -= QoF_ij
            end
        end

        if k < size(U,2)
            UR = U[:,k + 1]
        else
            # UR = U[:,1] 
            UR = params.BC(U[:,end], params.domain.ub, t)
        end

        Uf_k = params.operators.Vf * U[:,k]
        Uf_ext_k = params.operators.Vf_ext * [UL; UR]
        boundary_terms = (params.operators.nrJ' .* params.operators.L) * params.f_surface.(Uf_k, Uf_ext_k, params.operators.nrJ)

        forcing_terms = params.operators.Pq * params.forcing.(params.operators.Vq*U[:,k], xq_k, t)

        dUdt[:,k] =  - params.operators.M \ vol_terms .- boundary_terms .+ forcing_terms
        UL = U[:,k]
    end

    if params.use_SRD
        SRD!(dUdt, U, params.operators)
    end
end

function rhs(U, t, params)
    dUdt = similar(U)
    rhs!(dUdt, U, t, params)
    return dUdt
end

function get_entropy_variables(U, equations)
    V = similar(U)

    for k in axes(U,2)
        V[:,k] = cons2entropy.(U[:,k], equations)
    end

    return V
end

function get_entropy_residual(dUdt, U, equations, params)
    V = get_entropy_variables(U, equations)

    entropy_residual = 0.0
    for k in axes(U, 2)
        scaling_k = get_operator_scaling(params.dx)
        entropy_residual += scaling_k * V[:,k]' * params.operators.M * dUdt[:,k]
    end

    return entropy_residual
end

function get_mass(U, params)
    mass = zero(eltype(U))
    ones_vec = ones(length(U[:,1]))
    for k in axes(U,2)
        scaling_k = get_operator_scaling(params.dx)
        mass += scaling_k * ones_vec' * params.operators.M * U[:,k]
    end
    return mass
end


function get_cartesian_indices(x, num_elems, domain; edge_tol=1e-12)
    dx = (domain.ub - domain.lb) / num_elems
    index_real = (x - domain.lb) ./ dx

    index0 = floor.(Int64, index_real)
    on_boundary = abs(index0 - index_real) < edge_tol
    if !on_boundary || index0 == 0 # x is inside an element or at the lower boundary
        return index0 + 1
    elseif on_boundary && index0 == num_elems
        return index0
    else # x is on an interior boundary and therefore touches two elements
        return [index0, index0+1]
    end
end

function CutDGSolution(p, num_elems, cuts, domain; elem_type=Float64)
    dx = (domain.ub - domain.lb) / num_elems
    cuts_sorted = sort(cuts)

    cart_dof = zeros(elem_type, p+1, num_elems)
    iscut = zeros(Bool, num_elems)
    cut_dof = [ Vector{SVector{3,Float64}}[] for i in 1:num_elems]
    for i in eachindex(cuts_sorted) 
        I_cut = get_cartesian_indices(cuts_sorted[i], num_elems, domain)
        if length(I_cut) == 1 # The cut is inside an element
            if !iscut[i]
                push!(cut_dof[I_cut], zeros(elem_type, p+1))
                iscut[I_cut] = true
            end
            push!(cut_dof[I_cut], zeros(elem_type, p+1))
        end
    end
    
    dx = (domain.ub .- domain.lb) ./ num_elems
    cut_bounds = [ Bounds{1, Float64}[] for k in 1:num_elems ]
    i_cut = 1
    for k in 1:num_elems
        if iscut[k]
            cart_bounds_k = get_element_bounds(domain.lb, dx, k)
            x_lb = cart_bounds_k.lb
            for i_subelement in 1:length(cut_dof[k])-1
                push!(cut_bounds[k], Bounds(x_lb, cuts[i_cut]))
                x_lb = cuts[i_cut]
                i_cut += 1
            end

            push!(cut_bounds[k], Bounds(x_lb, cart_bounds_k.ub))
        end
    end
    return CutDGSolution(cart_dof, cut_dof, cuts_sorted, iscut, cut_bounds, domain, p, num_elems)
end

function setIC!(U::CutDGSolution, IC, operators)
    dx = (U.domain.ub - U.domain.lb) / U.n_elem
    for k in 1:U.n_elem
        if !U.iscut[k] # cartesian element
            bounds_k = get_element_bounds(U.domain.lb, dx, k)
            xq_k = ref2phys(operators.rq, bounds_k)
            U.cart_dof[:,k] = operators.Pq * IC.(xq_k, bounds_k.ub)
        else # cut element 
            for i in eachindex(U.cut_dof[k])
                bounds_k_i = U.cut_bounds[k][i]
                xq_k = ref2phys(operators.rq, bounds_k_i)
                U.cut_dof[k][i] = operators.Pq * IC.(xq_k, bounds_k_i.ub)
            end
        end
    end
end

# If:
# wet_cut element is small:
#     1) d(h(wet_cut))/dt > 0 (boundary expanding: wet cut element has gained mass)
#        => Apply double-sided SRD to the wet-cut element
#     2) d(h(wet_cut))/dt < 0 && d(h(dry_cut))/dt > 0 (boundary expanding: wet cut element has lost mass to its dry neighbor)
#        => Apply double-sided SRD
#     3) d(h(wet_cut))/dt < 0 && d(h(dry_cut))/dt == 0 (boundary contracting: wet cut element has lost mass to its wet neighbor)
#        => Apply one-sided SRD (on the wet-side of the boundary)
# dry_cut element is small:
#     1) d(h(dry_cut))/dt > 0 (boundary expanding: dry element has gained mass)
#        => Apply double-sided SRD
#     2) d(h(dry_cut))/dt == 0 (boundary non-expanding)
#        => Do nothing
function SRD(dUdt::CutDGSolution, U::CutDGSolution, operators; vol_threshold=0.45)
    # Find the cut elements
    # If both the wet and cut elements are "large", do not apply SRD
    # The wet cut element is small:
    # The dry cut element is small and has been given mass:
end

function SRD_doublesided(UL, U, UR, boundsL, bounds, boundsR, operators)
    return UL_new, U_new, UR_new
end

function SRD_singlesided(U, U_nbr, bounds, bounds_nbr, operators)
end

function Base.:+(UL::CutDGSolution, UR::CutDGSolution)
    return U_sum
end


# Moved from CutDG.jl:

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


function Base.:+(UL::CutDGSolution, UR::CutDGSolution)
    # TODO: update the boundary here 
    # Reshape the solution to the right of the leftmost cut

    if UL.p == 0 # Use adaptive penalty weighting for the FV case

    end
    return U_sum
end
