include("./End2End.jl")
using .End2End

F,∂F,params = setup(lb=0.11,ub=0.85,
                    filename="alldat_tio2_NIST_9wavs.dat",
                    ncells=1160,
                    npix=240,nintg=4,
                    ntruth=50,
                    nthr=2,
                    freqs=vec([1.09 1.07 1.04 1.02 1.0 0.98 0.96 0.94 0.92]),
                    δx = 0.91, δy = 0.91,
                    Dz = 2292.74,
                    ϵ_free = 1.0, μ_free = 1.0,
                    ntrain = 4,
                    α = 2000.0);
p₀ = ((params.lb+params.ub)/2) .* ones(params.ncells*params.ncells)
optrun!(p₀, F,∂F,params);
