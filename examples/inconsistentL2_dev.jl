using LinearAlgebra
using Plots
using Printf

using StartUpDG

include("../src/CutDG.jl")
using .CutDG

include("../src/TimeIntegration.jl")
using .TimeIntegration

include("solver_helper_functions.jl")

# Construct operators
p = 2;
rd = RefElemData(Line(), SBP{TensorProductLobatto}(), p)
Q = rd.M * rd.Dr

Vf_ext = zeros(size(rd.Vf,1), 2*(size(rd.Vf,2)))
Vf_ext[1,   1:size(rd.Vf,2)]     = rd.Vf[end,:]
Vf_ext[end, size(rd.Vf,2)+1:end] = rd.Vf[1,:]

operators = (; M=rd.M, Q, L=rd.LIFT, Pq=rd.Pq, r=rd.r, rq=rd.rq, Vq=rd.Vq, rf=rd.rf, Vf=rd.Vf, Vf_ext, nrJ=floor.(Int64,sign.(rd.nrJ)), VDM=rd.VDM, )

# The strong solution:
domain = Bounds(0.0, 1.0);


# For lag = 0, freq must be in [0.5, Inf)
freq = 1.0;
lag = 0.0;
x0 = 0.5 / freq + lag;
u_true(x) = x < x0 ? cos(freq*pi*(x-lag)) : 0.0;
subdomain_true = Bounds(0.0, x0)

# Plotting parameters
dx = 0.001;
x_plot = domain.lb:dx:domain.ub;
true_soln_lw = 7.5;
cut_lw = 2.5;
L2_lw = 5;

cut_color=:deepskyblue
L2_color=:red


M_mixed_true = get_mixed_mass_matrix(subdomain_true, domain, operators);
moments_true = M_mixed_true * u_true.(ref2phys(rd.rq, subdomain_true));



x_cut = 0.35;
n = 6;
plot(
    # legend=:outerbottom, 
    legend=false,
    size=(1500, 1200),
    xlim=(domain.lb, domain.ub),
    ylim=(-0.25, 1.5),
    xtickfontsize=24,
    ytickfontsize=24,
    legendfontsize=24
)


dx_cut = -0.1;
for i in 1:n
    x_cut_i = 0.95 + dx_cut*(i-1)
    subdomain_inconsistent = Bounds(0.0, x_cut_i);
    M_inconsistent = get_mixed_mass_matrix(subdomain_inconsistent, domain, operators);
    u_inconsistent = M_inconsistent \ moments_true;

    plot!(ref2phys(rd.rp, subdomain_inconsistent), rd.Vp*u_inconsistent, label="", color=cut_color, linewidth=cut_lw)
    vline!([x_cut_i], color=:blue, label="")
end

# # The proper L2 projection:
# M_proper = get_mixed_mass_matrix(domain, domain, operators);
# u_proper = M_proper \ moments_true;

plot!(ref2phys(rd.rp, domain), rd.Vp*u_proper, label="Proper L2 Projection", color=:red, linewidth=L2_lw)
vline!([domain.ub], color=:red, label="")
# png("figures/inconsistentL2_consistentL2.png")


plot!(x_plot, u_true.(x_plot), label="True Solution (x0=$x0)", color=:black, linewidth=true_soln_lw)
vline!([x0], color=:black, label="", linewidth=2.5)
hline!([0.0], color=:black, label="", linewidth=2.5)
# png("figures/inconsistentL2_true_soln.png")


# The inconsistent L2 projection:
subdomain_inconsistent = Bounds(0.0, x_cut);
M_inconsistent = get_mixed_mass_matrix(subdomain_inconsistent, domain, operators);
u_inconsistent = M_inconsistent \ moments_true;

plot!(ref2phys(rd.rp, subdomain_inconsistent), rd.Vp*u_inconsistent, label="Inconsistent L2 Projections", color=:blue, linewidth=6)
vline!([x_cut], color=:blue, label="")

png("figures/inconsistentL2_$x_cut.png")

# Test the penalization routine: