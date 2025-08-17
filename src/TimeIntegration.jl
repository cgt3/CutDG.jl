module TimeIntegration

export ForwardEuler, SSPRK33, SSPRK43

struct RKTableau
    c::AbstractArray
    a::AbstractArray
    b::AbstractArray

    function RKTableau(c, a, b)
        if length(c) != length(b) - 1
            throw(ErrorException("RKTableau: Number of nodes/time coefficients ($(length(c))) does not match number of stages ($(length(b)))."));
        elseif size(a, 1) != length(b) - 1 || size(a,2) != length(b) - 1
            throw(ErrorException("RKTableau: Size of RK matrix ($(size(a))) does not match number of stages ($(length(b)))"))
        end

        return new(c,a,b);
    end
end

# NOTE: the row index in a and the index of c are off by one to save memory
function (tableau::RKTableau)(U, t, dt::Float64, rhs, params)
    k = typeof(U)[];
    push!(k, rhs(U, t, params));

    # Special case the forward Euler step
    U_new = U + tableau.b[1]*dt * k[1];

    # All other intermediate states use the RK matrix and nodes
    for i in 2:length(tableau.b)
        U_temp = U;
        for j in 1:i-1
            U_temp += tableau.a[i-1,j]*dt * k[j]; 
        end

        k_i = rhs(U_temp, t + tableau.c[i-1]*dt, params);
        push!(k, k_i);

        U_new = U_new + tableau.b[i]*dt * k_i;
    end

    return U_new;
end

const ForwardEuler = RKTableau([], Matrix{Float64}(undef, (0,0)), [1]);
const SSPRK33 = RKTableau([1.0, 0.5], [1.0 0.0; 0.25 0.25], [1.0/6, 1.0/6, 2.0/3]);
const SSPRK43 = RKTableau([0.5, 1.0, 0.5], [0.5 0.0 0.0; 0.5 0.5 0.0; 1.0/6 1.0/6 1.0/6], [1.0/6, 1.0/6, 1.0/6, 0.5]);



end