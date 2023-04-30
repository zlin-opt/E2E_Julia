import Pkg; Pkg.add("FastChebInterp")
import Pkg; Pkg.add("ThreadsX")
import Pkg; Pkg.add("Zygote")

using DelimitedFiles
using FastChebInterp
using ThreadsX
using Base.Threads
using Zygote

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

function evalmodel(model::Any,p::Vector{Float64})
    ThreadsX.map(a->chebjacobian(model,a),p)
end

function r2c(t)
    nrows = size(t)[1]
    ncols = size(t[1][1])[1]÷2
    F = Array{ComplexF64,2}(undef,nrows,ncols)
    ∂F = Array{ComplexF64,2}(undef,nrows,ncols)
    Threads.@threads for i in 1:nrows
        F[i,:]  = t[i][1][1:2:end] + im * t[i][1][2:2:end]
        ∂F[i,:] = t[i][2][1:2:end] + im * t[i][2][2:2:end]
    end
    (F,∂F)
end

"""
    eval2c

Multi-threaded evaluation of meta-atom transmission coefficients 
for multiple frequencies using the chebyshev "model". 
Known benchmark: 9 million 5 freqs (=90 mil evals) of a 1000-degree poly take 4 sec on 64 threads
"""
function eval2c(model::cheb,p::Vector{Float64})::c∂tup{2}
    ndof = size(p)[1]
    nfreqs = size(model(p[1]))[1]÷2
    F = Array{ComplexF64,2}(undef,ndof,nfreqs)
    ∂F = Array{ComplexF64,2}(undef,ndof,nfreqs)
    Threads.@threads for i in 1:ndof
        @inbounds t = chebjacobian(model,p[i])
        @inbounds F[i,:]  = t[1][1:2:end] + im * t[1][2:2:end]
        @inbounds ∂F[i,:] = t[2][1:2:end] + im * t[2][2:2:end]
    end
    (F,∂F)
end

function end2end(model::FastChebInterp.ChebPoly, p::Vector{Float64}, getF::c∂tup{2}, f::Float64, fdat::Any)::r∂tup
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

x = rand(lb:0.000000001:ub,1000000)
@time map(a->chebjacobian(model,a),x);
@time ThreadsX.map(a->chebjacobian(model,a),x);
x = rand(lb:0.000000001:ub,1000000)
@time map(a->chebjacobian(model,a),x);
@time ThreadsX.map(a->chebjacobian(model,a),x);
x = rand(lb:0.000000001:ub,1000000)
@time map(a->chebjacobian(model,a),x);
@time ThreadsX.map(a->chebjacobian(model,a),x);


p = rand(lb:0.000000001:ub,1000000)
@time evalmodel(model,p);
p = rand(lb:0.000000001:ub,1000000)
@time evalmodel(model,p);
p = rand(lb:0.000000001:ub,1000000)
@time evalmodel(model,p);

println("hello")

lb,ub=0.11,0.68
filename="alldat_5wavs.dat"
model = getmodel(lb,ub,filename)
p = rand(lb:0.000000001:ub,1000000)
@time eval2c(model,p)[2][4435,:];
p = rand(lb:0.000000001:ub,1000000)
@time eval2c(model,p)[2][5131,:];
p = rand(lb:0.000000001:ub,1000000)
@time eval2c(model,p)[1][1155,:];
println("hello")

model(0.51198)

a = ones(Float64,3,3)
b = 2*ones(Float64,3,3)
sum(a.*b,dims=2)

function f(F::Array{ComplexF64,2})::Float64
    #sum(abs2.(F))
    sum(real.(F.^2) + 2.0*imag.(F))
end

function makeF(p::Vector{Float64})::c∂tup{2}
    F = reduce(hcat,(1+2im)*[p,p,p])
    tmp = ones(size(p))
    ∂F = reduce(hcat,(1+2im)*[tmp,tmp,tmp])
    (F,∂F)
end

function compute(p::Vector{Float64},makeF::c∂tup{2},f::Float64)::r∂tup
    F,∂F = makeF(p)
    ret,back = Zygote.pullback(f,F)
    (ret, real.(sum(conj.(back(1)[1]).*∂F, dims=2))[:,1])
end
    
p=[1.,2.,3.]
@time compute(p,makeF,f)
p=rand(10)
@time a = compute(p,makeF,f)
typeof(a)


