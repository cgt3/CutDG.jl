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

function initialize_solution(p, num_elems)

    return U
end