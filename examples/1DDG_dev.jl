# In this example: ==================================================================
# System: SWE
#   - Problem:
#       - Domain: [-1, 1]
#       - t = 0, ..., 1
#       - ICs: Wet dambreak
#       - BCs: Wall (zero-normal velocity)
#   - Fluxes:
#       - Volume flux: Wintermeyer
#       - Surface flux: Lax-Friedrichs
#   - Time Integration: Forward Euler/SSPRK33 or 43
#   - Mesh: Fixed, uniform 1D Cartesian mesh
#       - No adaptivity (single level, single patch mesh)
#       - No cuts (static or dynamic)
# ===================================================================================

using LinearAlgebra
using Plots
using Printf

using StartUpDG
using Trixi
using TrixiShallowWater

# include("../src/CutDG.jl")
# using .CutDG

include("../src/TimeIntegration.jl")
using .TimeIntegration

include("solver_helper_functions.jl")

# Problem parameters ==============================================================================
equations = ShallowWaterEquations1D(gravity=1.0)
domain = Bounds(-3.0, 3.0)

t_start = 0.0;
t_end = 1.0;

# Wall (zero-normal velocity) BCs:
function BC(U, x, t) 
    U_ghost = similar(U)
    n = length(U)
    for i in eachindex(U_ghost)
        i_rev = n - i + 1
        U_ghost[i] = SVector(U[i_rev][1], -U[i_rev][2], U[i_rev][3])
    end
    return U_ghost
end

# Zero forcing (constant bathymetry)
forcing(U, x, t) = SVector(0.0, 0.0, 0.0);

# Wet dam break IC:
x0 = 0.0;
h0_downstream = 5.0;
h0_upstream = h0_downstream + 1.0;
IC(x; left_eval=false) = x < x0 || (left_eval && x <= x0) ? SVector(h0_upstream, 0.0, 0.0) : SVector(h0_downstream, 0.0, 0.0);

# # Gaussian pulse IC:
# x0 = 0.0;
# w = 1.0
# h0 = 3.0;
# IC(x; left_eval=false) =SVector(h0 + exp(-*(x-x0)^2/(w^2)), 0.0, 0.0);


# Formulation: ====================================================================================
# Set the discretization parameters
p = 3;     # Degree of the DG solution
nx = 10;   # Number of elements
dt = 1e-2; # Time step 

dx = (domain.ub - domain.lb) / nx;

t_all = t_start:dt:t_end;
n_t = length(t_all);


# Construct the formulation operators
rd = RefElemData(Line(), SBP{TensorProductLobatto}(), p)
Q = rd.M * rd.Dr

Vf_ext = zeros(size(rd.Vf,1), 2*(size(rd.Vf,2)))
Vf_ext[1,   1:size(rd.Vf,2)]     = rd.Vf[end,:]
Vf_ext[end, size(rd.Vf,2)+1:end] = rd.Vf[1,:]

operators = (; M=rd.M, Q, L=rd.LIFT, Pq=rd.Pq, rq=rd.rq, Vq=rd.Vq, rf=rd.rf, Vf=rd.Vf, Vf_ext, nrJ=floor.(Int64,sign.(rd.nrJ)), VDM=rd.VDM, )

# Set the numerical fluxes
f_volume(UL, UR)  = flux_wintermeyer_etal(UL, UR, 1, equations);
f_surface(UL, UR, orientation) = orientation == 1 ? flux_lax_friedrichs(UL, UR, 1, equations) : flux_lax_friedrichs(UR, UL, 1, equations); 
# f_surface(UL, UR, orientation) = flux_lax_friedrichs(UL, UR, 1, equations)


# Allocate memory for the DG solution
x = zeros(p + 1, nx);
elem_type = SVector{3, Float64}
U = zeros(elem_type, p+1, nx)

for k in axes(x,2)
    bounds_k = get_element_bounds(domain.lb, dx, k)
    x[:, k] = ref2phys(rd.r, bounds_k)

    xq_k = ref2phys(rd.rq, bounds_k)
    if xq_k[end] == x0
        U[:, k] = rd.Pq*IC.(xq_k, left_eval=true)
    else
        U[:, k] = rd.Pq*IC.(xq_k)
    end
end


params = (; domain, operators, dx, f_volume, f_surface, BC, forcing)


# # Sanity Check: Plot the basis vectors
# plot_basis_vectors(rd)

# Sanity Check: Plot the IC
# display(plot_DG_solution(U, rd, domain, ylims=(0,5)))


# # Sanity Check: Check that dUdt=0 for a lake at rest
dUdt = zeros(elem_type, p+1, nx)
rhs!(dUdt, U, 0.0, params)


# Forward Euler time steps
entropy_residual = zeros(Float64, n_t)
mass = zeros(eltype(U), n_t)
@gif for i_t in 1:n_t
    global U, dUdt

    if i_t % 10 == 0
        println("i_t = $i_t")
    end
    rhs!(dUdt, U, t_all[i_t], params)

    entropy_residual[i_t] = get_entropy_residual(dUdt, U, equations, params)
    mass[i_t] = get_mass(U, params)

    # U = U + dt * dUdt
    # U = ForwardEuler(U, dt*i_t, dt, rhs, params)
    U = SSPRK33(U, dt*i_t, dt, rhs, params)
    
    plot_DG_solution(U, rd, domain, ylims=(0, 8))
end

# # Validation: Entropy Residual
# max_entropy_res = maximum(entropy_residual)
# entropy_plot_title = @sprintf("Entropy Residual (max. value=%.4e)", max_entropy_res)
# plot(t_all, entropy_residual, title=entropy_plot_title, legend=false, xlabel="t", linewidth=4)
# hline!([0], color=:black)


# Validation: Conservation
plot(title="Conservation: Mass", xlabel="t")
plot!(t_all, getindex.(mass,1) .- mass[1][1], legend=false, linewidth=3)
hline!([0], color=:black)

# # Note: wall BCs do not conserve momentum; use periodic BCs if momentum conservation is desired
# plot(title="Conservation: Momentum", xlabel="t")
# plot!(t_all, getindex.(mass,2) .- mass[1][2], legend=false, linewidth=3)
# hline!([0], color=:black)
