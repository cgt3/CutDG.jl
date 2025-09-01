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

const REF_DOMAIN = Bounds(-1.0, 1.0)

struct CutDGSolution{dim, order}
    cart_dof
    cut_dof
    cuts
    iscut
    isactive
    cut_bounds
    domain
    p
    order
    n_elem
    operators
end


function CutDGSolution(p, num_elems, cuts, domain, operators; elem_type=Float64)
    dim = 1 # TODO: extend to 2/3D
    order = 0 # 0th order time derivative

    dx = (domain.ub - domain.lb) / num_elems
    cuts_sorted = sort(cuts)

    cart_dof = zeros(elem_type, p+1, num_elems)
    iscut = zeros(Bool, num_elems)
    isactive = [ [false] for i in 1:num_elems]
    cut_dof = [ Vector{SVector{3,Float64}}[] for i in 1:num_elems]
    for i in eachindex(cuts_sorted) 
        I_cut = get_cartesian_indices(cuts_sorted[i], num_elems, domain)
        if length(I_cut) == 1 # The cut is inside an element
            if !iscut[i]
                push!(cut_dof[I_cut], zeros(elem_type, p+1))
                iscut[I_cut] = true
            end
            push!(isactive[I_cut], false)
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

    return CutDGSolution{dim, order}(cart_dof, cut_dof, cuts_sorted, iscut, isactive, cut_bounds, domain, p, order, num_elems, operators)
end

# For allocating memory for cut solution with the same mesh/cuts as the given U
function CutDGSolution(U::CutDGSolution, order)
    dim = 1 # TODO: extend to 2/3D
    cart_dof_new = 0.0 .* U.cart_dof
    isactive_new = false .&& U.isactive
    return CutDGSolution{dim, order}( cart_dof_new, 0.0 .* U.cut_dof, U.cuts, U.iscut, isactive_new, U.cut_bounds, U.domain, U.p, order, U.n_elem, U.operators)
end

function isactive(dof)
    return any(dof .!= zeros(eltype(dof), size(dof)))
end

function is_active_or_adjacent(U, UL, UR, isactive)
    return isactive(U) || isactive(UL) || isactive(UR)
end

function has_active(U::CutDGSolution, k)
    return sum(U.isactive[k]) > 0
end


function getvelocity(U)
    return U[1] > 0 ? U[2] / U[1] : 0.0
end

function momentumFlux(U; g=1.0)
    v = getvelocity(U)
    return U[2] * v + 0.5*U[1]^2 * g
end

function max_abs_speed_naive_modified(UL, UR; g=1.0)
    # Get the velocity quantities
    vL = getvelocity(UL)
    vR = getvelocity(UR)

    # Calculate the wave celerity on the left and right
    cL = sqrt(g * UL[1])
    cR = sqrt(g * UR[2])

    return max(abs(vL), abs(vR)) + max(cL, cR)
end

function cons2entropy(U; g=1.0)
    h, _, b = U

    v = getvelocity(U)

    w1 = g * (h + b) - 0.5f0 * v^2
    w2 = v

    return SVector(w1, w2, 0.0)
end

function flux_central_modified(UL, UR; g=1.0)
    vL = getvelocity(UL)
    vR = getvelocity(UR)

    f_h  = 0.5*(UL[2] + UR[2])
    f_hu = 0.5*(momentumFlux(UL) + momentumFlux(UR))

    return SVector(f_h, f_hu, 0.0)
end

function flux_lax_friedrichs_modified(UL, UR; g=1.0, onesided_tol=1e-12, dry_tol=1e-14)
    f = flux_central_modified(UL, UR, g=g) - max_abs_speed_naive_modified(UL, UR, g=g) * (UR - UL)

    # Do not let dry elements "give" mass
    if ( UR[1] < dry_tol && f[1] < -onesided_tol ) || ( UL[1] < dry_tol && f[1] > onesided_tol )
        return SVector(0.0, 0.0, 0.0)
    end

    return f
end

# TODO: may need to be adjusted to be one-sided?
function flux_wintermeyer_etal_modified(UL, UR; g=1.0)
    # Unpack left and right state
    hL, huL, _ = UL
    hR, huR, _ = UR

    # Get the velocities on either side
    vL = getvelocity(UL)
    vR = getvelocity(UR)

    # Average each factor of products in flux
    v_avg = 0.5f0 * (vL + vR)
    p_avg = 0.5f0 * g * hL * hR

    # Calculate fluxes depending on orientation
    f1 = 0.5f0 * (huL + huR)
    f2 = f1 * v_avg + p_avg

    return SVector(f1, f2, 0)
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

function getfirst(U::CutDGSolution, k)
    if U.iscut[k]
        return U.cut_dof[k][1]
    else
        return U.cart_dof[:, k]
    end
end

function getlast(U::CutDGSolution, k)
    if U.iscut[k]
        return U.cut_dof[k][end]
    else
        return U.cart_dof[:, k]
    end
end

function getnext(U::CutDGSolution, k)
    if U.iscut[k+1]
        return U.cut_dof[k+1][1]
    else
        return U.cart_dof[:, k+1]
    end
end

function getnext(U::CutDGSolution, k, i)
    if i < length(U.cut_dof[k])
        return U.cut_dof[k][i+1]
    else
        return getnext(U, k)
    end
end

function getprev(U::CutDGSolution, k)
    if U.iscut[k-1]
        return U.cut_dof[k-1][end]
    else
        return U.cart_dof[:,k]
    end
end

function getprev(U::CutDGSolution, k, i)
    if i > 1
        return U.cut_dof[k][i-1]
    else
        return getprev(U, k)
    end
end


function compute_rhs_single(U, UL, UR, bounds, t, params)
    vol_terms = similar(U)
    boundary_terms = similar(U)

    xq_k = ref2phys(params.operators.rq, bounds)
        
    # Calculate volume terms
    vol_terms .*= 0;
    for i in axes(params.operators.Q,1)
        for j in 1:i 
            QoF_ij = 2 * params.operators.Q[i,j] * params.f_volume(U[i], U[j])
            vol_terms[i] += QoF_ij
            vol_terms[j] -= QoF_ij
        end
    end

    # Calculate boundary terms
    Uf_k = params.operators.Vf * U
    Uf_ext_k = params.operators.Vf_ext * [UL; UR]
    boundary_terms = (params.operators.nrJ' .* params.operators.L) * params.f_surface.(Uf_k, Uf_ext_k, params.operators.nrJ)

    # Forcing
    forcing_terms = params.operators.Pq * params.forcing.(params.operators.Vq*U, xq_k, t)

    return - params.operators.M \ vol_terms .- boundary_terms .+ forcing_terms
end

function rhs!(dUdt::CutDGSolution, U::CutDGSolution, t, params; isactive=isactive)
    UL = params.BC(U.cart_dof[:,1], params.domain.lb, t)
    for k in 1:U.n_elem
        if !U.iscut[k] # Cartesian element
            bounds_k = get_element_bounds(params.domain.lb, params.dx, k)
            if k < U.n_elem
                UR = getnext(U, k)
            else
                UR = params.BC(U.cart_dof[:,end], params.domain.ub, t)
            end

            if is_active_or_adjacent(U.cart_dof[:,k], UL, UR, isactive)
                dUdt.cart_dof[:,k] = compute_rhs_single(U.cart_dof[:,k], UL, UR, bounds_k, t, params)
                dUdt.isactive[k][1] = isactive(dUdt.cart_dof[:,k])
            end
            UL = U.cart_dof[:,k]
        else # Cut element
            for i in eachindex(U.cut_dof[k])
                bounds_k_i = U.cut_bounds[k][i]
                if i < length(U.cut_dof[k]) || k < U.n_elem
                    UR = getnext(U, k, i)
                else
                    UR = params.BC(U.cut_dof[k][i], params.domain.ub, t)
                end

                if is_active_or_adjacent(U.cut_dof[k][i], UL, UR, isactive)
                    dUdt.cut_dof[k][i] = compute_rhs_single(U.cut_dof[k][i], UL, UR, bounds_k_i, t, params)
                    dUdt.isactive[k][i] = isactive(dUdt.cut_dof[k][i])
                end
                UL = U.cut_dof[k][i]
            end
        end
    end

    # NOTE: SRD will need to change isactive in some cases (double-sided nbhds for small dry cells)
    # if params.use_SRD
    #     SRD!(dUdt, U, params.operators)
    # end
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
    dUdt = CutDGSolution(U, 1)
    rhs!(dUdt, U, t, params)
    return dUdt
end

function get_entropy_variables(U::CutDGSolution; g=1.0)
    V = CutDGSolution(U.p, U.n_elem, U.cuts, U.domain, U.operators, elem_type=SVector{3, Float64})
    for k in eachindex(U.iscut)
        if U.iscut[k]
            for i in eachindex(U.cut_dof[k])
                V.cut_dof[k][i] = cons2entropy.(U.cut_dof[k][i], g=g)
            end
        else
            V.cart_dof[:,k] = cons2entropy.(U.cart_dof[:,k], g=g)
        end
    end

    return V
end

function get_entropy_variables(U; g=1.0)
    V = similar(U)

    for k in axes(U,2)
        V[:,k] = cons2entropy.(U[:,k], g=g)
    end

    return V
end

function get_entropy_residual(dUdt::CutDGSolution, U::CutDGSolution, params; g=1.0)
    V = get_entropy_variables(U, g=g)

    entropy_residual = 0.0
    for k in 1:U.n_elem
        if U.iscut[k] 
            for i in eachindex(U.cut_dof[k])
                scaling_k_i = get_operator_scaling(U.cut_bounds[k][i])
                entropy_residual += scaling_k_i * V.cut_dof[k][i]' * params.operators.M * dUdt.cut_dof[k][i]
            end
        else
            scaling_k = get_operator_scaling(params.dx)
            entropy_residual += scaling_k * V.cart_dof[:,k]' * params.operators.M * dUdt.cart_dof[:,k]
        end
    end

    return entropy_residual
end

function get_entropy_residual(dUdt, U, params; g=1.0)
    V = get_entropy_variables(U, g=g)

    entropy_residual = 0.0
    for k in axes(U, 2)
        scaling_k = get_operator_scaling(params.dx)
        entropy_residual += scaling_k * V[:,k]' * params.operators.M * dUdt[:,k]
    end

    return entropy_residual
end

function get_mass(U::CutDGSolution, params)
    mass = zero(eltype(U.cart_dof))
    ones_vec = ones(length(U.cart_dof[:,1]))
    for k in 1:U.n_elem
        if U.iscut[k]
            for i in eachindex(U.cut_dof[k])
                scaling_k_i = get_operator_scaling(U.cut_bounds[k][i])
                mass += scaling_k_i * ones_vec' * params.operators.M * U.cut_dof[k][i]
            end
        else
            scaling_k = get_operator_scaling(params.dx)
            mass += scaling_k * ones_vec' * params.operators.M * U.cart_dof[:,k]
        end
    end
    return mass
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

function setIC!(U::CutDGSolution, IC, isactive, operators)
    dx = (U.domain.ub - U.domain.lb) / U.n_elem
    for k in 1:U.n_elem
        if !U.iscut[k] # Cartesian element
            bounds_k = get_element_bounds(U.domain.lb, dx, k)
            xq_k = ref2phys(operators.rq, bounds_k)
            U.cart_dof[:,k] = operators.Pq * IC.(xq_k, bounds_k.ub)
            U.isactive[k][1] = isactive(U.cart_dof[:,k])
        else # cut element 
            for i in eachindex(U.cut_dof[k])
                bounds_k_i = U.cut_bounds[k][i]
                xq_k = ref2phys(operators.rq, bounds_k_i)
                U.cut_dof[k][i] = operators.Pq * IC.(xq_k, bounds_k_i.ub)
                U.isactive[k][i] = isactive(U.cut_dof[k][i])
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

function get_element_bounds(U::CutDGSolution, k)
    dx = (U.domain.ub - U.domain.lb) / U.n_elem
    return Bounds(U.domain.lb + dx*(k-1), U.domain.lb + dx*k)
end

function get_operator_scaling(bounds::Bounds; ref_domain=Bounds(-1.0, 1.0))
    return (bounds.ub - bounds.lb) / (ref_domain.ub - ref_domain.lb)
end

function get_operator_scaling(dx::Real; ref_bounds=Bounds(-1.0, 1.0))
    return dx / (ref_bounds.ub - ref_bounds.lb)
end

function get_elem_mass(bounds::Bounds, U, operators; ref_domain=Bounds(-1.0, 1.0))
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

function get_superbounds(bounds1::Bounds, bounds2::Bounds)
    return Bounds(min(bounds1.lb, bounds2.lb), max(bounds1.ub, bounds2.ub))
end

function get_intersection(bounds1::Bounds, bounds2::Bounds)
    return Bounds(max(bounds1.lb, bounds2.lb), min(bounds1.ub, bounds2.ub))
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


function get_new_boundary(domain_orig, moments_orig, uf_ext_L, operators; dx_tol=1e-6, penalty_weights=[0.5, 0.5], ref_domain=Bounds(-1.0, 1.0))
    # Find the moments of u_orig
    # moments_orig = get_operator_scaling(domain_orig) * operators.M * u_orig 
    uf_true = [uf_ext_L, 0.0]

    # For checking positivity:
    dr_dense = 1e-4
    r_dense = ref_domain.lb:dr_dense:ref_domain.ub
    V_pos = vandermonde(Line(), length(operators.r)-1, r_dense) / operators.VDM

    # Reduce the interval size until the new solution is positive
    penalty_min = Inf
    u_opt = similar(moments_orig)

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
                penalty = penalty_weights' * abs.(operators.Vf*u_new .- uf_true)
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

function Base.:*(c::Real, U::CutDGSolution)
    cU = CutDGSolution(U.p, U.n_elem, U.cuts, U.domain, U.operators, elem_type=eltype(U.cart_dof))
    for k in eachindex(U.iscut)
        if U.iscut[k] 
            cU.cart_dof[:,k] = 0.0*U.cart_dof[:,k]
            for i in eachindex(U.cut_dof[k])
                cU.cut_dof[k][i] = c*U.cut_dof[k][i]
            end
        else
            cU.cart_dof[:,k] = c * U.cart_dof[:,k]
        end
    end

    return cU
end

function get_partial_element(U::CutDGSolution, bounds::Bounds)
    m_all = get_moments_all(U, bounds)

    return U.operators.M \ m_all / get_operator_scaling(bounds)
end

# TODO: double Check
function get_moments(U, bounds_U, bounds, operators; ref_domain=Bounds(-1.0, 1.0))
    # If the intervals are disjoint, return 0
    if bounds_U.ub < bounds.lb || bounds_U.lb > bounds.ub
        return zeros(eltype(U), length(U))
    end

    # Find the intersection of the two intervals, which by definition will be a subset of U_bounds
    subbounds = get_intersection(bounds_U, bounds)

    # Convert U to the basis on 'bounds':
    # 1) Map r of subbounds to r of bounds_U
    r_subbounds_lb = (ref_domain.ub-ref_domain.lb) * (subbounds.lb - bounds_U.lb) / (bounds_U.ub - bounds_U.lb) + ref_domain.lb
    r_subbounds_ub = (ref_domain.ub-ref_domain.lb) * (subbounds.ub - bounds_U.lb) / (bounds_U.ub - bounds_U.lb) + ref_domain.lb
    subbounds_r = Bounds(r_subbounds_lb, r_subbounds_ub)
    r_subbounds_U = ref2phys(operators.r, subbounds_r)

    # 2) Evaluate U at r_subbounds
    U_subbounds = (vandermonde(Line(), length(U) - 1, r_subbounds_U) / operators.VDM) * U 


    # Compute the moments of U on subbounds using the mixed mass matrix as normal
    M_bounds = get_mixed_mass_matrix(subbounds, bounds, operators)
    return M_bounds * U_subbounds
end

function get_moments_all(U::CutDGSolution, bounds)
    dx = (U.domain.ub - U.domain.lb) / U.n_elem

    I_lb = maximum(get_cartesian_indices(bounds.lb, U.n_elem, U.domain))
    I_ub = minimum(get_cartesian_indices(bounds.ub, U.n_elem, U.domain))

    k_min = min(I_lb, I_ub)
    k_max = max(I_lb, I_ub)
    println("bounds=[$(bounds.lb), $(bounds.ub)], k_min=$k_min, k_max=$k_max")

    m = zeros(eltype(U.cart_dof), size(U.cart_dof, 1))
    for k in k_min:k_max 
        if U.iscut[k]
            for i in eachindex(U.cut_dof[k])
                moments = get_moments(U.cut_dof[k][i], U.cut_bounds[k][i], bounds, U.operators)
                m += moments
            end
        else
            bounds_k = get_element_bounds(U.domain.lb, dx, k)
            m += get_moments(U.cart_dof[:,k], bounds_k, bounds, U.operators)
        end
    end

    return m
end

# TODO: adapt to having multiple cuts, differing p, etc etc
function Base.:+(U1::CutDGSolution, U2::CutDGSolution)
    # Error checking
    if U1.domain != U2.domain
        throw(ErrorException("+(CutDGSolution, CutDGSolution): Cannot add solutions on different domains."))
    elseif U1.n_elem != U2.n_elem # TODO: Support in the future?
        throw(ErrorException("+(CutDGSolution, CutDGSolution): Cannot add solutions with different background meshes."))
    elseif U1.p != U2.p # TODO: Support in the future?
        throw(ErrorException("+(CutDGSolution, CutDGSolution): Cannot add solutions of different order."))
    end

    # Allocate memory and assign known variables
    operators = U1.operators

    p = U1.p 
    n_elem = U1.n_elem
    domain = U1.domain
    dx = (domain.ub - domain.lb) / n_elem

    cart_dof = zeros(eltype(U1.cart_dof), p+1, n_elem)
    iscut = zeros(Bool, n_elem)

    isactive = [ [false] for i in 1:n_elem]

    cut_dof = [ Vector{SVector{3,Float64}}[] for i in 1:n_elem]
    cut_bounds = [ Bounds{1, Float64}[] for k in 1:n_elem ]

    # Direcly add common (active) Cartesian elements
    k = 1
    while k < n_elem && !U1.iscut[k] && U1.isactive[k][1]==1 && !U2.iscut[k] && U2.isactive[k][1]==1
        cart_dof[:,k] .= U1.cart_dof[:, k] + U2.cart_dof[:,k]
        push!(isactive[k], true)
        k += 1
    end
    k_cart_end = k-1
    k_transition_start = k 

    # Find the end of the transition region
    k_max_active = k
    while k_max_active < n_elem && ( has_active(U1, k_max_active) || has_active(U2, k) )
        k_max_active += 1
    end

    # Find the Cartesian upperbound of the transition region
    bounds_k_min = get_element_bounds(U1, k_cart_end)
    bounds_k_max = get_element_bounds(U1, k_max_active)
    bounds_transition = Bounds(bounds_k_min.ub, bounds_k_max.ub)

    # Compute the moments of the solution in the transition region:
    moments1_trans = get_moments_all(U1, bounds_transition)
    moments2_trans = get_moments_all(U2, bounds_transition)
    moments_trans = moments1_trans + moments2_trans

    # TODO: make sure mass is positive in the transition region

    # Find the new boundary in the transition region:
    Uf_ext_L = operators.Vf_ext * [cart_dof[:, k_cart_end]; zeros(eltype(U1.cart_dof), p+1)]
    # Find the new cut position using the mass variable
    pos_field = 1
    _, x_cut_new, _ =  get_new_boundary(bounds_transition, getindex.(moments_trans, pos_field), getindex(Uf_ext_L[1], pos_field), U1.operators)
    
    # Reshape the solution using the new cut position
    bounds_trans_new = Bounds(bounds_transition.lb, x_cut_new)
    M_transition = get_mixed_mass_matrix(bounds_trans_new, bounds_transition, operators)
    display(moments_trans)
    U_transition_new = inv(M_transition) * moments_trans

    # Redistribute the new solution onto the background mesh
    I_cut = get_cartesian_indices(x_cut_new, n_elem, domain)
    L_ref = REF_DOMAIN.ub - REF_DOMAIN.lb
    L_trans = bounds_trans_new.ub - bounds_trans_new.lb
    if length(I_cut) == 1 # The cut is inside an element
        for k in k_transition_start:I_cut - 1
            # Interpolate the transition solution to this element
            bounds_k = get_element_bounds(U1, k)
            r_k = (bounds_k.ub - bounds_k.lb) / L_trans * (operators.r .- REF_DOMAIN.lb)  .+ REF_DOMAIN.lb;

            # Construct the Vandermonde matrix of the superbasis evaluated at the subdomain's 
            # nodes in the superdomain
            V_k = vandermonde(Line(), rd.N, r_sub) / operators.VDM

            cart_dof = V_k * U_transition_new
            push!(isactive[k], true)
        end

        # Interpolate the transition solution to this element
        k = I_cut
        iscut[k] = true
        bounds_k_cart = get_element_bounds(U1, k)
        bounds_k = Bounds(bounds_k_cart.lb, x_cut_new)
        r_k = (bounds_k.ub - bounds_k.lb) / L_trans * (operators.r .- REF_DOMAIN.lb)  .+ REF_DOMAIN.lb;
        V_k = vandermonde(Line(), rd.N, r_k) / operators.VDM

        # For the wet cut element
        push!(cut_dof[k], V_k * U_transition_new)
        push!(cut_bounds[k], bounds_k)
        push!(isactive[k], true)

        # For the dry cut element
        push!(cut_dof[k], zeros(eltype(U_transition_new), length(U_transition_new)))
        push!(cut_bounds[k], Bounds(x_cut_new, bounds_k_cart.ub))
        push!(isactive[k], false)

    else # The cut is conforming
        k_max = I_cut[1]
        for k in k_transition_start:k_max
            # Interpolate the transition solution to this element
            bounds_k = get_element_bounds(U1, k)
            r_k = (bounds_k.ub - bounds_k.lb) / L_trans * (operators.r .- REF_DOMAIN.lb)  .+ REF_DOMAIN.lb;

            # Construct the Vandermonde matrix of the superbasis evaluated at the subdomain's 
            # nodes in the superdomain
            V_k = vandermonde(Line(), rd.N, r_k) / operators.VDM

            cart_dof = V_k * U_transition_new
            push!(isactive[k], true)
        end
    end

    return CutDGSolution{1, 0}(cart_dof, cut_dof, [x_cut_new], iscut, isactive, cut_bounds, domain, p, 0, n_elem, operators)
end

# TODO/BUG: pushing zeros to the solution fields will not be correct for non-constant bathymetry

# # TODO: adapt to having multiple cuts, differing p, etc etc
# function Base.:+(U1::CutDGSolution, U2::CutDGSolution)
#     # Error checking
#     if U1.domain != U2.domain
#         throw(ErrorException("+(CutDGSolution, CutDGSolution): Cannot add solutions on different domains."))
#     elseif U1.n_elem != U2.n_elem # TODO: Support in the future?
#         throw(ErrorException("+(CutDGSolution, CutDGSolution): Cannot add solutions with different background meshes."))
#     elseif U1.p != U2.p # TODO: Support in the future?
#         throw(ErrorException("+(CutDGSolution, CutDGSolution): Cannot add solutions of different order."))
#     end

#     # Allocate memory and assign known variables
#     operators = U1.operators

#     p = U1.p 
#     n_elem = U1.n_elem
#     domain = U1.domain
#     dx = (domain.ub - domain.lb) / n_elem

#     cart_dof = zeros(eltype(U1.cart_dof), p+1, n_elem)
#     iscut = zeros(Bool, n_elem)

#     cut_dof = [ Vector{SVector{3,Float64}}[] for i in 1:n_elem]
#     cut_bounds = [ Bounds{1, Float64}[] for k in 1:n_elem ]

#     # Identify transition regions
#     is_transition_elem = zeros(Bool, n_elem)
#     for k in 1:n_elem 
#         println("k=$k: has_active(U1)=$(has_active(U1,k)), has_active(U2)=$(has_active(U2,k))")
#         is_transition_elem[k] = has_active(U1, k) != has_active(U2, k)
#     end

#     return is_transition_elem

#     # In each transition region, propogate boundaries individually

#     # Create a supermesh of the new boundaries

#     # Merge elements on the supermesh to go back to a conforming cut mesh
# end
