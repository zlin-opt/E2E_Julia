include("./End2End.jl")
using .End2End

F,∂F,params = setup(lb=0.11,ub=0.68,
                    filename="alldat_20wavs.dat",
                    ncells=5000,
                    npix=600,nintg=5,
                    ntruth=120,
                    nthr=4,
                    freqs=vec([1.25 1.22 1.2  1.17 1.14 1.12 1.09 1.07 1.04 1.01 0.99 0.96 0.93 0.91 0.88 0.86 0.83 0.8  0.78 0.75]),
                    δx = 0.79, δy = 0.79,
                    Dz = 10000.0,
                    ϵ_free = 1.0, μ_free = 1.0,
                    ntrain = 4,
                    α = 100.0);
p₀ = ((params.lb+params.ub)/2) .* ones(params.ncells*params.ncells)
optrun!(p₀, F,∂F,params);
