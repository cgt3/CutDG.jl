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
#   - Time Integration: Forward Euler
#   - Mesh: Fixed, uniform 1D Cartesian mesh
#       - No adaptivity (single level, single patch mesh)
#       - No cuts (static or dynamic)
# ===================================================================================

using LinearAlgebra
using Plots

using StartUpDG
using Trixi
using TrixiShallowWater

include("../src/CutDG.jl")
using .CutDG

# Problem parameters ==============================================================================
equations = ShallowWaterEquations1D(gravity=1.0)
domain = Bounds(-3.0, 3.0)

t_end = 1.0;
t_span = (0.0, t_end)

# Wall (zero-normal velocity) BCs:
BC(U, x, t, params) = SVector(U[1], -U[2], U[3]);

# Wet dam break IC:
x0 = 0.0;
h0_downstream = 1.0;
h0_upstream = h0_downstream + 1.0;
IC(x; left_eval=false) = x < x0 || (left_eval && x <= x0) ? SVector(h0_upstream, 0.0, 0.0) : SVector(h0_downstream, 0.0, 0.0);

# Formulation: ====================================================================================
# Set the discretization parameters
p = 2;     # Degree of the DG solution
nx = 10;   # Number of elements
dt = 1e-2; # Time step 

dx = (domain.ub - domain.lb) / nx;


# Construct the formulation operators
rd = RefElemData(Line(), SBP{TensorProductLobatto}(), p)
op_scaling = get_operator_scaling(domain);

# Set the numerical fluxes
f_volume(UL, UR)  = flux_wintermeyer_etal(UL, UR, 1, equations);
f_surface(UL, UR) = flux_lax_friedrichs(UL, UR, 1, equations);

# Allocate memory for the DG solution
x = zeros(p + 1, nx);
elem_type = SVector{3, Float64}
U = zeros(elem_type, p+1, nx)

for k in axes(x,2)
    bounds_k = get_element_bounds(domain.lb, dx, k)
    x[:, k] = ref2phys(bounds_k, rd.r)

    xq_k = ref2phys(bounds_k, rd.rq)
    if xq_k[end] == x0
        U[:, k] = rd.Pq*IC.(xq_k, left_eval=true)
        println("Element $k")
    else
        U[:, k] = rd.Pq*IC.(xq_k)
    end
end


# Sanity Check: Plot the basis vectors
basis_vec_plot = plot(xlim=(-1, 1), title="Basis Vectors")
e_i = zeros(p+1)
for i in 1:p+1
    e_i[i] = 1
    plot!(basis_vec_plot, rd.rp, rd.Vp * e_i, label="phi_$i")
    e_i[i] = 0
end
display(basis_vec_plot)

# Sanity Check: Plot the IC
field = 1
IC_plot = plot(xlim=(domain.lb, domain.ub), ylim=(0,3), title="IC: p=$p, nx=$nx", legend=false)
for k in 1:nx
    bounds_k = get_element_bounds(domain.lb, dx, k)
    plot!(IC_plot, ref2phys(bounds_k, rd.rp), rd.Vp * getindex.(U[:,k],field), color=:blue)
    vline!(IC_plot, [bounds_k.lb], color=:grey80)
end
display(IC_plot)





