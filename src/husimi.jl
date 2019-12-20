export husimi

using LinearAlgebra # for ⋅ (cdot)

struct Zero end
Base.:+(x, ::Zero) = x
Base.convert(::Type{Float64}, ::Zero) = 0.0
Base.convert(::Type{Any}, ::HusimiFunctions.Zero) = 0.0

const VSV{D} = Vector{SVector{D, Float64}}

"""
    husimi(ψ, r::Vector{<:SVector}, r₀::SVector, k₀::SVector, σ[, B]; thread)
Calculate the Husimi function of `ψ` at `(r₀, k₀)` with precision `σ`.
`ψ` is a vector whose values `ψ[i]` is the wavefunction value
at position `r[i]` (`r` is a vector of `SVector`s).
The keyword `thread` means to parallelize the operation over processor threads
(`true` by default).

Optionally calculate in a non-zero magnetic field `B`.
The units of `r, σ` are expected in nm, `k` in 1/nm and B in Tesla.

Notice that the magnetic transformation assumes Landau gauge in x orientation,
see https://arxiv.org/abs/1912.04622 for defining other gauges (by changing
the function `magneticphase`).
"""
function husimi(ψ, r, r₀::SVector, k₀::SVector, σ, B = Zero();
                thread = true)
    if thread
        abs2(coherent_projection_threaded(ψ, r, r₀, k₀, σ, B))
    else
        abs2(coherent_projection(ψ, r, r₀, k₀, σ, B))
    end
end

##############################################################
# Projection
##############################################################
# TODO: Check if dot is as optimized as  r[1]*r[1] + r[2]*r[2]
function coherent_projection_threaded(ψ, r, r₀, k₀, σ, B = Zero())
    D = length(r₀)
    sums = zeros(ComplexF64, Threads.nthreads())
    skip = (3*σ)^2
    denominator = (4*σ^2)

    @inbounds @fastmath Threads.@threads for i in 1:length(r)
        δr = r[i] - r₀
        rsq = δr⋅δr
        rsq > skip && continue
        space = exp(-rsq/denominator)
        phase = k₀⋅δr
        mphase = magneticphase(r[i], r₀, B)
        ss, cs = sincos(phase + mphase)
        sums[Threads.threadid()] += conj(ψ[i]) * space * complex(cs, ss)
    end
    return sum(sums)/(σ*sqrt(2π))^(D/2)
end

function coherent_projection(ψ, r, r₀, k₀, σ, B = Zero())
    D = length(r₀)
    sums = zero(ComplexF64)
    skip = (3*σ)^2
    denominator = (4*σ^2)

    @inbounds @fastmath for i in 1:length(r)
        δr = r[i] - r₀
        rsq = δr⋅δr
        rsq > skip && continue
        space = exp(-rsq/denominator)
        phase = k₀⋅δr
        mphase = magneticphase(r[i], r₀, B)
        ss, cs = sincos(phase + mphase)
        sums += conj(ψ[i]) * space * complex(cs, ss)
    end
    return sums/(σ*sqrt(2π))^(D/2)
end

const prefac = (10.0^(-2.0))/6.582 # e/ħ prefactor, when multiplying Tesla gives 1/nm^2

magneticphase(ri, r₀, B::Real) =
@inbounds prefac * B/2 * (ri[1]*ri[2] - r₀[1]*ri[2] + ri[1]*r₀[2])
magneticphase(ri, r₀, B::Zero) = Zero()
