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

function plot_DG_solution(U, rd, domain; ylims=(0.0,3.0), field=1)
    field = 1
    IC_plot = plot(xlim=(domain.lb, domain.ub), ylim=ylims, title="IC: p=$p, nx=$nx", legend=false)
    for k in axes(U,2)
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
