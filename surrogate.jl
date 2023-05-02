import Pkg; 
Pkg.add("FastChebInterp");
Pkg.add("ThreadsX");
Pkg.add("Zygote");
Pkg.add("Memoize");

using DelimitedFiles
using FastChebInterp
using ThreadsX
using Base.Threads
using Zygote
using BenchmarkTools
using Memoize

const c∂tup{N} = Tuple{Array{ComplexF64,N},Array{ComplexF64,N}}
const r∂tup = Tuple{Float64,Vector{Float64}}
const cheb = FastChebInterp.ChebPoly

"""
    getmodel

Generates a chebyshev polynomial interpolated from the datafile. 
The latter must be in the format ipt, DoF, Re(t[freq1]), Im(t[freq2]) ... 
In other words, the dimensions of the datafile must be (order+1,2+2*nfreqs)
"""
function getmodel(lb,ub,filename)
    dat = readdlm(filename,' ',Float64,'\n')
    dat = dat[:,3:end]'
    dat = [dat[:,i] for i in 1:size(dat,2)]
    model = chebinterp(dat,lb,ub)
end

"""
    eval2c

Multi-threaded evaluation of meta-atom transmission coefficients 
for multiple frequencies using the chebyshev "model". 
"""

@memoize function wrapcheb(model::cheb,p::Float64)
    chebjacobian(model,p)
end

function eval2c(model::cheb,p::Vector{Float64};moi::Bool=false)::c∂tup{2}
    ndof = size(p)[1]
    nfreqs = size(model(p[1]))[1]÷2
    F = Array{ComplexF64,2}(undef,ndof,nfreqs)
    ∂F = Array{ComplexF64,2}(undef,ndof,nfreqs)
    Threads.@threads for i in 1:ndof
        if moi==true
            @inbounds t = wrapcheb(model,p[i])
        else
            @inbounds t = chebjacobian(model,p[i])
        end
        @inbounds @views F[i,:]  .= t[1][1:2:end] .+ im * t[1][2:2:end]
        @inbounds @views ∂F[i,:] .= t[2][1:2:end] .+ im * t[2][2:2:end]
    end
    (F,∂F)
end

function end2end(model::cheb, p::Vector{Float64}, getF::Function, f::Function, fdat::Any)::r∂tup
    F,∂F = getF(model,p)
    ret,back = Zygote.pullback(ξ->f(ξ,fdat),F)
    (ret, real.(sum(conj.(back(1)[1]).*∂F, dims=2))[:,1])
end


# setup(;ncells::Int64=3000,npix::Int64=500,nintg::Int64=5,nspatl::Int64=120,
#        Dz::Float64=5000, freqs::Vector{Float64}=[1.2,1.1,1.0,0.9,0.8],
#        lb::Float64=0.11,ub::Float64=0.68,
#        filename::String="alldat_5wavs.dat", 
#        kwargs...)

lb,ub=0.11,0.68
filename="alldat_5wavs.dat"
model = getmodel(lb,ub,filename)
(F,∂F) = @btime eval2c(model,rand(lb:0.0000000001:ub,9000000),moi=false);
(F,∂F) = @btime eval2c(model,rand(lb:0.0000000001:ub,9000000),moi=true);