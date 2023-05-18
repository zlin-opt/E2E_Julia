module End2End

import Pkg; 
Pkg.add("FastChebInterp");
Pkg.add("Zygote");
Pkg.add("BenchmarkTools");
Pkg.add("FiniteDifferences");
Pkg.add("FFTW");
Pkg.add("PaddedViews");
Pkg.add("ChainRules");
Pkg.add("ChainRulesCore");
Pkg.add("LinearMaps");
Pkg.add("IterativeSolvers");
Pkg.add("NLopt")

using DelimitedFiles
using FastChebInterp
using Base.Threads
using Zygote
using BenchmarkTools
using FiniteDifferences
using LinearAlgebra
using FFTW
using PaddedViews
using ChainRulesCore
using Random
using LinearMaps
using IterativeSolvers
using NLopt
Random.seed!(1234)

export getmodels, eval2c!, end2end!, create_2Dffp_plans, fftconv2d, green2d, ffp, near2far, F2G, G2Û!, Û2G!, G2ℓ, ParamsE2E, setup, F2ℓ, optrun!

function getmodels(lb,ub,filename)
    
    dat = readdlm(filename,' ',Float64,'\n')
    nfreqs = (size(dat,2)-2)÷2
    models = Vector{FastChebInterp.ChebPoly{1,ComplexF64,Float64}}(undef,nfreqs)
    Threads.@threads for i in 1:nfreqs
        models[i] = chebinterp(complex.(dat[:,3+2*(i-1)],dat[:,4+2*(i-1)]),lb,ub)
    end
    models
end

function eval2c!(F,∂F, models::AbstractVector{<:FastChebInterp.ChebPoly},p::AbstractVector)
    println("Evaluating the transmission coefficients"); flush(stdout)
    Threads.@threads for c in CartesianIndices(F)
        i,j = Tuple(c)
        F[c],∂F[c] = chebgradient(models[j],p[i])
    end
end

function end2end!(gdat, F,∂F, models, p, f, fdat, iter_print=[0,10])
    eval2c!(F,∂F, models,p)
    ret,back = Zygote.pullback(ξ->f(ξ,fdat),F)
    gdat[:] .= real.(vec(sum(conj.(back(1)[1]) .* ∂F, dims=2)))
    println("step $(iter_print[1]) returns $ret"); flush(stdout)
    if mod(iter_print[1],iter_print[2])==0 
        writedlm("p@step$(iter_print[1]).txt",p)
    end
    iter_print[1] += 1
    return ret
end

function create_2Dffp_plans(n::Int64,nthr::Int64)
    FFTW.set_num_threads(nthr)
    plan_fft(rand(ComplexF64, n,n),flags=FFTW.MEASURE)
end

function fftconv2d(arr::Array{ComplexF64,2},fftker::Array{ComplexF64,2},plan,fwd::Bool=true)::Array{ComplexF64,2}
    narr = size(arr)[1]
    nker = size(fftker)[1]
    nout = nker - narr

    i1 = fwd == true ? 1+(narr÷2) : narr÷2
    i2 = i1 + nout -1
    ifftshift(plan \ (fftker .* (plan * collect(sym_paddedviews(0.0+im*0.0,arr,fftker)[1]))))[i1:i2,i1:i2]
    
end
    
function green2d(nx::Int64,ny::Int64, δx::Float64, δy::Float64, freq::Float64, ϵ::Float64,μ::Float64, Dz::Float64, plan)::Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}

    ω = 2*π*freq
    n = sqrt(ϵ*μ)
    k = n*ω
    ik = im * k

    Lx,Ly = nx*δx, ny*δy
    δxy = δx*δy

    x = range(-Lx/2,Lx/2-δx, nx)'
    y = range(-Ly/2,Ly/2-δy, ny)

    gz = @. Dz * (-1 + ik * sqrt(x^2 + y^2 + Dz^2)) * cis(k*sqrt(x^2 + y^2 + Dz^2))/(4*π*sqrt(x^2 + y^2 + Dz^2)^3) * δxy * (-μ/ϵ) * (1/sqrt(ω))
    
    fgz = plan * gz
    fgzT = plan * reverse(gz)

    (fgz,fgzT)

end

# Field matrix F has the format (unit cells, frequencies)
function ffp(F::Array{ComplexF64,2}, fgs::Vector{Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}}, plan, fwd::Bool)::Array{ComplexF64,2}
    
    narr = Int64(sqrt(size(F)[1]))
    nker = size(fgs[1][1])[1]
    nout = nker - narr
    nfreqs = size(F)[2]

    out = zeros(ComplexF64, nout*nout, nfreqs)
    Threads.@threads for i in 1:nfreqs
        println("Performing FFP for freq $i"); flush(stdout)
        @inbounds out[:,i] .= vec(fftconv2d( reshape(F[:,i],narr,narr),
                                             fwd==true ? fgs[i][1] : fgs[i][2],
                                             plan, fwd ))
    end
    out
end

function near2far(F::Array{ComplexF64,2}, fgs::Vector{Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}}, plan)::Array{ComplexF64,2}
    ffp(F,fgs,plan, true)
end
    

function ChainRulesCore.rrule(::typeof(near2far), F::Array{ComplexF64,2}, fgs::Vector{Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}}, plan)
    efar = near2far(F,fgs,plan)
    function near2far_pullback(vec::Array{ComplexF64,2})

        dF = @thunk(ffp(conj.(vec), fgs, plan, false))
        NoTangent(), conj.(dF), ZeroTangent(), ZeroTangent()

    end
    efar, near2far_pullback
end

function F2G(F,nintg)
    
    ncells=Int64(sqrt(size(F)[1]))
    npix = ncells÷nintg
    nfreqs=size(F)[2]
    sum(reshape(abs2.(F),(nintg,npix,nintg,npix,nfreqs)),dims=(1,3))[1,:,1,:,:]

end

function g2g̃(g)
    g̃ = complex.(zeros(size(g)),zeros(size(g)))
    n = size(g)[3]
    Threads.@threads for i in 1:n
        g̃[:,:,i] .= fft(g[:,:,i])
    end
    g̃
end

function g2g̃ᵀ(g)
    g̃ᵀ = complex.(zeros(size(g)),zeros(size(g)))
    n = size(g)[3]
    Threads.@threads for i in 1:n
        g̃ᵀ[:,:,i] .= fft(reverse(g[:,:,i]))
    end
    g̃ᵀ
end

⊛(g̃,u) = real.(ifftshift( ifft( g̃ .* fft(u) ) ))
✪(v,u) = real.(ifftshift( ifft( fft(v) .* conj.(fft(u)))))
syz(u,g̃) = collect(sym_paddedviews(0,u,g̃)[1])
v∂gu(g̃,u,v) = ✪(syz(v,g̃),syz(u,g̃))

function conv(u,K̃,fwd::Bool)
    nᵢ, nₖ = size(u)[1], size(K̃)[1]
    nₒ = nₖ-nᵢ

    i₁ = fwd==true ? 1+(nᵢ÷2) : nᵢ÷2
    i₂ = i₁+nₒ-1
    ⊛(K̃,syz(u,K̃))[i₁:i₂,i₁:i₂]
end

function G(u,g̃)
    nᵢ, nₖ = size(u)[1], size(g̃)[1]
    nₒ = nₖ-nᵢ
    n = size(g̃)[3]
    Gu = zeros(Float64,nₒ,nₒ,n)
    Threads.@threads for i in 1:n
        Gu[:,:,i] .= conv(u[:,:,i],g̃[:,:,i],true)
    end

    sum(Gu,dims=3)[:,:,1]
end

function GᵀG(u,g̃,g̃ᵀ)
    
    Gu = G(u,g̃)
    n = size(g̃)[3]
    GᵀGu = zeros(size(u))
    Threads.@threads for i in 1:n
        GᵀGu[:,:,i] .= conv(Gu,g̃ᵀ[:,:,i],false)
    end

    GᵀGu
end

function y∂GᵀGx(g̃,x,y)

    Gx = G(x,g̃)
    Gy = G(y,g̃)

    n = size(g̃)[3]
    ∂g = zeros(size(g̃))
    Threads.@threads for i in 1:n
        ∂g[:,:,i] .= v∂gu(g̃[:,:,i],x[:,:,i],Gy) .+ v∂gu(g̃[:,:,i],y[:,:,i],Gx)
    end
    
    ∂g
end

GᵀGαI(x,g̃,g̃ᵀ,α) = GᵀG(x,g̃,g̃ᵀ) .+ α.*x

function regress(g̃,g̃ᵀ,α,y)

    dims = size(y)

    A(ξ) = vec(GᵀGαI(reshape(ξ,dims),g̃,g̃ᵀ,α))
    Â = LinearMap(A,size(vec(y))[1],issymmetric=true,isposdef=true)

    println("Entering CG ..."); flush(stdout)
    ret = reshape(cg(Â,vec(y),reltol=10^(-16)),dims)
    println("CG has finished."); flush(stdout)

    ret

end

function G2Û!(g,α,U,g̃,g̃ᵀ)

    g̃[:,:,:] = g2g̃(g)[:,:,:]
    g̃ᵀ[:,:,:] = g2g̃ᵀ(g)[:,:,:]
    
    Û = similar(U)
    Threads.@threads for i in 1:size(U)[4]
        println("Reconstruction stage: ground truth $i"); flush(stdout)
        y = GᵀG(U[:,:,:,i],g̃,g̃ᵀ)
        Û[:,:,:,i] .= regress(g̃,g̃ᵀ,α,y)[:,:,:]
    end
    Û

end

function Û2G!(∇ℓ,g̃,g̃ᵀ,α,U,Û)

    (n1,n2,n3) = size(g̃)
    n4 = size(U)[4]
    ∂g = zeros(Float64,n1,n2,n3,n4)
    Threads.@threads for i in 1:size(Û)[4]
        println("Backpropagation stage: ground truth $i"); flush(stdout)
        Λ = regress(g̃,g̃ᵀ,α,∇ℓ[:,:,:,i])
        ∂g[:,:,:,i] .= y∂GᵀGx(g̃,U[:,:,:,i].-Û[:,:,:,i],Λ)[:,:,:]
    end
    sum(∂g,dims=4)[:,:,:,1]

end

function ChainRulesCore.rrule(::typeof(G2Û!), g,α,U,g̃,g̃ᵀ)
    Û = G2Û!(g,α,U,g̃,g̃ᵀ)
    function G2Û!_pullback(∇ℓ)

        ∂g = @thunk(Û2G!(∇ℓ,g̃,g̃ᵀ,α,U,Û))
        NoTangent(), ∂g, ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent()

    end
    Û, G2Û!_pullback
end

function G2ℓ(g,U,α)

    g̃ = Array{ComplexF64,3}(undef,size(g))
    g̃ᵀ = Array{ComplexF64,3}(undef,size(g))
    Û = G2Û!(g,α,U,g̃,g̃ᵀ)

    ntrain = size(U)[4]
    ret = 0.0
    for i in 1:ntrain
        ret += norm(U[:,:,:,i].-Û[:,:,:,i])/norm(U[:,:,:,i])
    end

    ret/ntrain  
end

struct ParamsE2E
    lb::Float64
    ub::Float64
    models
    ncells::Int64
    ndof::Int64
    npix::Int64
    nintg::Int64
    ntruth::Int64
    npsf::Int64
    nfar::Int64
    ngreen::Int64
    nfreqs::Int64
    freqs::Vector{Float64}
    fgs::Vector{Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}}
    ffp_plan
    U::Array{Float64,4}
    α::Float64
end

function setup(;lb,ub,filename,ncells,npix,nintg,ntruth,nthr,freqs,δx,δy,Dz,ϵ_free,μ_free,ntrain,α)
    
    models = getmodels(lb,ub,filename)

    ndof = ncells * ncells

    npsf = ntruth + npix
    nfar = npsf * nintg
    ngreen = ncells + nfar
    nfreqs = size(freqs,1)
    ffp_plan = create_2Dffp_plans(ngreen,nthr)
    
    tmp_fgs = [ green2d(ngreen,ngreen, δx,δy, freqs[i], ϵ_free,μ_free, Dz,ffp_plan) for i in 1:nfreqs ]
    tmp = sqrt(mean(F2G(near2far(complex.(ones(ndof,nfreqs),zeros(ndof,nfreqs)), tmp_fgs, ffp_plan),nintg)))
    fgs = [ tmp .* tmp_fgs[i] for i in 1:nfreqs ]

    U = rand(ntruth,ntruth,nfreqs,ntrain)

    F = Array{ComplexF64,2}(undef,ncells*ncells,nfreqs)
    ∂F = Array{ComplexF64,2}(undef,ncells*ncells,nfreqs)
    params = ParamsE2E(lb,ub,models,ncells,ndof,npix,nintg,ntruth,npsf,nfar,ngreen,nfreqs,freqs,fgs,ffp_plan,U,α)
    
    (F,∂F,params)
end

function F2ℓ(F,params)
    efar = near2far(F, params.fgs, params.ffp_plan)
    g = F2G(efar,params.nintg)
    G2ℓ(g,params.U,params.α)
end    

function optrun!(p₀, F,∂F,params)

    ndof = size(p₀,1)

    opt = Opt(:LD_MMA, ndof)
    opt.lower_bounds = params.lb .* ones(ndof)
    opt.upper_bounds = params.ub .* ones(ndof)
    opt.xtol_rel = 1e-4

    iter_print = [0, 10]
    opt.min_objective = (x,g) -> end2end!(g, F,∂F, params.models, x, F2ℓ, params, iter_print)

    (minf,minx,ret) = optimize(opt, p₀)

end

end