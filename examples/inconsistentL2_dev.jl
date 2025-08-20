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
p = 5;
rd = RefElemData(Line(), SBP{TensorProductLobatto}(), p)
Q = rd.M * rd.Dr

Vf_ext = zeros(size(rd.Vf,1), 2*(size(rd.Vf,2)))
Vf_ext[1,   1:size(rd.Vf,2)]     = rd.Vf[end,:]
Vf_ext[end, size(rd.Vf,2)+1:end] = rd.Vf[1,:]

operators = (; M=rd.M, Q, L=rd.LIFT, Pq=rd.Pq, r=rd.r, rq=rd.rq, Vq=rd.Vq, rf=rd.rf, Vf=rd.Vf, Vf_ext, nrJ=floor.(Int64,sign.(rd.nrJ)), VDM=rd.VDM, )

# The strong solution:
domain = Bounds(0.0, 1.0);

# Set the strong solution:

# Cosine:
# For lag = 0, freq must be in [0.5, Inf)
freq = 1.0;
lag = 0.0;
x0 = 0.5 / freq + lag;
subdomain_true = Bounds(0.0, x0)
u_true(x) = x < x0 ? cos(freq*pi*(x-lag)) : 0.0;
function_name="cos(pi*x)"

# # Step:
# x0 = 0.5
# subdomain_true = Bounds(domain.lb, x0)
# u_true(x) = x <= x0 ? 1 : 0;
# errorL = -0.0;
# function_name="Step Function"

# Plotting parameters
dx = 0.001;
x_plot = domain.lb:dx:domain.ub;
r_plot = phys2ref(x_plot, domain)
V_plot = vandermonde(Line(), p, r_plot) / operators.VDM

true_soln_lw = 7.5;
cut_lw = 2.5;
L2_lw = 5;

L2_color=:red

# Calculate the moments of the true solution wrt the domain's basis
M_mixed_true = get_mixed_mass_matrix(subdomain_true, domain, operators);
moments_true = M_mixed_true * u_true.(ref2phys(rd.rq, subdomain_true));


# Calculate the proper L2 projection:
M_proper = get_mixed_mass_matrix(domain, domain, operators);
u_proper = M_proper \ moments_true;

uf_ext_L = u_true(domain.lb)

u_incon_opt, x_cut_opt, penalty = get_new_boundary(domain, u_proper, uf_ext_L, operators)
subdomain_opt = Bounds(domain.lb, x_cut_opt)

# Calculate the inconsistent L2 projection:
x_cut = x0;
subdomain_inconsistent = Bounds(0.0, x_cut);
M_inconsistent = get_mixed_mass_matrix(subdomain_inconsistent, domain, operators);
u_inconsistent = M_inconsistent \ moments_true;



# Plotting:
plot(
    # legend=:outerbottom, 
    legend=false,
    size=(1500, 1200),
    xlim=(domain.lb, domain.ub),
    # ylim=(-0.25, 1.5),
    xtickfontsize=24,
    ytickfontsize=24,
    legendfontsize=24
)

# cut_color=:deepskyblue
# dx_plot = 0.1;
# for x_prev in 0.95:-dx_plot:x_cut + dx_plot #dx_plot:0.9
#     subdomain_prev = Bounds(0.0, x_prev);
#     M_prev = get_mixed_mass_matrix(subdomain_prev, domain, operators);
#     u_prev = M_prev \ moments_true;

#     plot!(ref2phys(r_plot, subdomain_prev), V_plot*u_prev, label="", color=cut_color, linewidth=3)
#     vline!([x_prev], color=:blue, label="")
# end

plot!(ref2phys(r_plot, domain), V_plot*u_proper, label="Proper L2 Projection", color=:red, linewidth=L2_lw)
vline!([domain.ub], color=:red, label="")


plot!(x_plot, u_true.(x_plot), label="True Solution (x0=$x0)", color=:black, linewidth=true_soln_lw)
vline!([x0], color=:black, label="", linewidth=2.5)
hline!([0.0], color=:black, label="", linewidth=2.5)


# plot!(ref2phys(rd.rp, subdomain_opt), rd.Vp*u_incon_opt, label="Optimal Inconsistent L2 Projection", color=:green, linewidth=6)
# vline!([x_cut_opt], color=:green, label="")


# plot!(ref2phys(r_plot, subdomain_inconsistent), V_plot*u_inconsistent, label="Inconsistent L2 Projections", color=:blue, linewidth=6)
plot!(ref2phys(r_plot, subdomain_inconsistent), V_plot*u_inconsistent, label="", color=:blue, linewidth=6)

vline!([x_cut], color=:blue, label="")

# png("figures/inconsistentL2_$(function_name)_p$(p)_proper.png")
png("figures/inconsistentL2_$(function_name)_p$(p)_x$(x_cut).png")


# # Test the penalization routine:
# uL_true = u_true(0) + errorL;
# uR_true = 0.0;

# # For testing positivity
# r_pos = LinRange(-1, 1, 1000)
# V_pos = vandermonde(Line(), p, r_pos) / operators.VDM;

# dx_test = 0.001;
# x_test = domain.lb + dx_test:dx_test:domain.ub;

# # alpha = 0.5;
# alpha = 1.0 #0.5*(1+alpha);
# penaltyL = zeros(length(x_test))
# penaltyR = zeros(length(x_test))
# # penalty_all = zeros(length(x_test), 6)
# I_pos = Int64[]
# I_neg = Int64[]
# i_neg = zeros(Int64, length(x_test))
# for i in eachindex(x_test)
#     subdomain_test = Bounds(0.0, x_test[i]);
#     M_test = get_mixed_mass_matrix(subdomain_test, domain, operators);
#     u_test = M_test \ moments_true;
#     uf = operators.Vf * u_test;

#     penaltyL[i] = abs(uf[1] - uL_true)
#     penaltyR[i] = abs(uf[2] - uR_true)

#     if all( V_pos * u_test .>= 0 )
#         push!(I_pos, i)
#     else
#         i_neg[i] = true
#         push!(I_neg, i)
#     end
# end
# penalty_total = alpha*penaltyL + (1-alpha)*penaltyR;
# # penalty_all[:,p+1] = penalty_total;


# i_min_pos = argmin(penalty_total[I_pos])
# i_min = I_pos[i_min_pos]
# x_min = x_test[i_min]
# I_neg_R = maximum(I_pos):length(x_test)


# plot(xlim=(domain.lb, domain.ub), 
#     ylim=(0,1.0), 
#     size=(1500, 600),
#     legend=:outertopright,
#     title="Penalty Function for $function_name, p=$p, a=$alpha\n\n",
#     xtickfontsize=18,
#     ytickfontsize=18,
#     legendfontsize=20,
#     titlefontsize=28
#     )
# plot!(x_test, penaltyL, color=:blue, label="Left Penalty", linewidth=4)
# plot!(x_test, penaltyR, color=:red, label="Right Penalty", linewidth=4)
# plot!(x_test, penalty_total, linewidth=8, label="Total Penalty")
# vline!([x0], color=:black, label="x0=$x0", linewidth=3)
# vline!([x_min], color=:red, label="x_min=$x_min", linestyle=:dash, linewidth=4)

# png("figures/penalty_$(function_name)_p$(p)_alpha$alpha.png")


# plot!(x_test, i_neg, fillrange=0.0, fillstyle=:/, color=:black, label="")

# png("figures/penalty_$(function_name)_p$(p)_blacked.png")



# # Plot all penalty curves
# plot_all = plot(xlim=(domain.lb, domain.ub),
#     ylim=(0,1.0), 
#     size=(900, 500),
#     title="Penalty Functions for $function_name, p=0,...5\n",
#     legend=:outertopright,
#     xtickfontsize=14,
#     ytickfontsize=14,
#     legendfontsize=14,
#     titlefontsize=20
# )
# colors = palette(:Set1_5)
# for i_p = 1:5
#     plot!(plot_all, x_test, penalty_all[:,i_p ], label="p=$(i_p - 1)", linewidth=4, color=colors[i_p])
# end
# vline!([x0], color=:black, label="x0")
# display(plot_all)
# png("figures/penalty_$(function_name)_all.png")
