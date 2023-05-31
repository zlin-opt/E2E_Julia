include("./End2End.jl")
using .End2End
using FiniteDifferences

ncells = 6
lb = 0.11
ub = 0.68
F,∂F,params = setup(lb=0.11,ub=0.68,
                    filename="alldat_5wavs.dat",
                    ncells=ncells,
                    npix=4,nintg=2,
                    ntruth=2,
                    nthr=1,
                    freqs=[0.8,1.0,1.2],
                    δx = 0.8, δy = 0.8,
                    Dz = 500.0,
                    ϵ_free = 1.0, μ_free = 1.0,
                    ntrain = 4,
                    α = 100.0);
p₀ = rand(1.2*lb:0.01/ncells^2:0.8*ub,ncells^2)
gdat = zeros(ncells*ncells)
tmp = zeros(ncells*ncells)
end2end!(gdat, F,∂F, params.models, p₀, F2ℓ, params);
gdat1 = grad(central_fdm(5,1), x->end2end!(tmp, F,∂F, params.models, x, F2ℓ, params), p₀)[1]
display(maximum(abs.(gdat .- gdat1))/mean(abs.(gdat)))


