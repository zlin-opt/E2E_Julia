import Pkg; 
Pkg.add("FastChebInterp");
Pkg.add("ThreadsX");
Pkg.add("Zygote");
Pkg.add("Memoize");
Pkg.add("BenchmarkTools");
Pkg.add("Memoize")

using DelimitedFiles
using FastChebInterp
using ThreadsX
using Base.Threads
using Zygote
using BenchmarkTools
using Memoize

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
    eval2c!(F,∂F, model,p)

In-place multi-threaded evaluation of meta-atom transmission coefficients for multiple frequencies using the chebyshev model. 

F and ∂F must be pre-allocated as
 F = Array{ComplexF64,2}(undef,#unit cells,#freqs)
∂F = Array{ComplexF64,2}(undef,#unit cells,#freqs)
"""
function eval2c!(F,∂F, model::cheb,p::Vector{Float64})
    ndof = size(p)[1]
    Threads.@threads for i in 1:ndof
        @inbounds t,∂t = chebjacobian(model,p[i])
        @inbounds @views @.  F[i,:] = complex( t[1:2:end], t[2:2:end])
        @inbounds @views @. ∂F[i,:] = complex(∂t[1:2:end],∂t[2:2:end])
    end
end

"""
Explanation: for f(z=x+iy) ∈ ℜ, Zygote returns df = ∂f/∂x + i ∂f/∂y 
The Wirtinger derivative is ∂f/∂z = 1/2 (∂f/∂x - i ∂f/∂y) = 1/2 conj(df)
The chain rule is ∂f/∂p = ∂f/∂z ∂z/∂p + ∂f/∂z' ∂z'/∂p = 2 real( ∂f/∂z ∂z/∂p ) = real( conj(df) ∂z/∂p ) 
Gradient vector gdat must be pre-allocated as
gdat = Vector{Float64}(undef,#unit cells)
"""
function end2end!(gdat, F,∂F, model::cheb, p::Vector{Float64}, getF!::Function, f::Function, fdat::Any)
    getF!(F,∂F, model,p)
    ret,back = Zygote.pullback(ξ->f(ξ,fdat),F)
    gdat[:] .= real.(sum(conj.(back(1)[1]) .* ∂F, dims=2))[:,1]
    return ret
end


# setup(;ncells::Int64=3000,npix::Int64=500,nintg::Int64=5,nspatl::Int64=120,
#        Dz::Float64=5000, freqs::Vector{Float64}=[1.2,1.1,1.0,0.9,0.8],
#        lb::Float64=0.11,ub::Float64=0.68,
#        filename::String="alldat_5wavs.dat", 
#        kwargs...)

lb,ub=0.11,0.68
filename="alldat_5wavs.dat"
model = getmodel(lb,ub,filename)
ncells = 10000000
p = rand(lb:0.01/ncells:ub,ncells)
F = Array{ComplexF64,2}(undef,ncells,5)
∂F = Array{ComplexF64,2}(undef,ncells,5)
@btime eval2c!($F,$∂F, $model, $p);

function f(F,fdat)
    sum(real.(F).*imag.(F).^2)
end

gdat = Vector{Float64}(undef,ncells)
@btime end2end!($gdat, $F,$∂F, $model,$p, $eval2c!, $f, Nothing)

import Pkg; Pkg.add("FiniteDifferences")

using FiniteDifferences
using LinearAlgebra
lb,ub=0.11,0.68
filename="alldat_5wavs.dat"
model = getmodel(lb,ub,filename)
ncells = 100
p = rand(1.2*lb:0.01/ncells:0.8*ub,ncells)
F = Array{ComplexF64,2}(undef,ncells,5)
∂F = Array{ComplexF64,2}(undef,ncells,5)
eval2c!(F,∂F, model, p);
gdat = Vector{Float64}(undef,ncells)
function f2(F,fdat)
    sum(real.(F).*imag.(F).^2)
end
end2end!(gdat, F,∂F, model,p, eval2c!, f2, Nothing)
tmp(x) = end2end!(gdat, F,∂F, model,x, eval2c!, f2, Nothing) 
Δ = grad(central_fdm(5,1), tmp, p)[1]
maximum( abs.(Δ .- gdat)[1:end-1] )/mean(Δ)

a = rand(3,3)
display(a)
sum(a,dims=2)[:,1]


