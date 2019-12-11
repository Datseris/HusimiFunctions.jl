using LinearAlgebra # for ⋅ (cdot)

struct Zero end
Base.:+(x, ::Zero) = x
Base.convert(Float64, ::Zero) = 0.0

const VSV{D} = Vector{SVector{D, Float64}}

struct Husimi{D}
    Q::Matrix{Float32}
    r0s::VSV{D}
    k0s::VSV{D}
    σ::Float64
    B::Float64
end

Husimi(Q::Matrix, r, k, σ) = Husimi(Q, r, k, σ, 0)
Husimi(Q::Matrix, r::VSV{D}, k, σ, B) where {D} = Husimi{D}(Q, r, k, σ, B)

"""
    Husimi(ψ::Vector{Complex}, r, r0s, k0s, σ, [B])
Calculate a `Husimi` function at positions `r0s` and wavevectors `k0s`,
for the wavefunction `ψ`. The resulting struct has the following fields:
```julia
Q, r0s, k0s, σ, B
```
where `Q[i,j]` is measured at `r0s[i], k0s[j]`
*notice that `ψ` is not stored*.

`r, r0s, k0s` must be vectors of `SVector`s.

    Husimi(Q::Matrix, r, r0s, k0s, σ, [B])
Directly instantiate the `Husimi` container with a pre-calculated `Q`.
"""
function Husimi(ψ, r, r0s::VSV{D}, k0s::VSV{D}, σ, B = Zero();
    views = [todo for i in 1:length(r0s)]) where {D}
    Q = zeros(Float32, length(r0s), length(k0s))
    for j in 1:length(k0s)
        for i in 1:length(r0s)
            @inbounds Q[i,j] = husimi(ψ, r, r0s[i], k0s[j])
        end
    end
    return Husimi{D}(Q, r0s, k0s, σ, B)
end

"""
    husimi(ψ, r::Vector{<:SVector}, r₀::SVector, k₀::SVector, σ[, B])
Calculate the Husimi function of `ψ` at `(r₀, k₀)` with precision `σ`.
`ψ` is a vector whose values `ψ[i]` is the wavefunction value
at position `r[i]` (`r` is a vector of `SVector`s).

Optionally calculate in a non-zero magnetic field `B`.

The units of `r, σ` are expected in nm, `k` in 1/nm and B in Tesla.
"""
function husimi(ψ, r, r₀::SVector, k₀::SVector, σ, B = Zero())
    abs2(coherent_projection(ψ, r, r₀, k₀, σ, B))
end

##############################################################
# Projection
##############################################################
# TODO: Check if dot is as optimized as  r[1]*r[1] + r[2]*r[2]
function coherent_projection(ψ, r, r₀, k₀, σ, B = Zero())
    D = length(r₀)
    sums = zeros(ComplexF64, Threads.nthreads())
    skip = (3*σ)^2
    denominator = (4*σ^2)

    @inbounds @fastmath Threads.@threads for i in 1:length(r)
        δr = r[i] - r₀
        rsq = δr⋅δr
        rsq > skip && continue
        # Projection to coherent state:
        space = exp(-rsq/denominator)
        phase = k₀⋅δr
        mphase = magneticphase(r[i], r₀, B)
        ss, cs = sincos(phase + mphase)
        sums[Threads.threadid()] += conj(ψ[i]) * space * complex(cs, ss)
    end
    return sum(sums)/(σ*sqrt(2π))^(D/2)
end

const prefac = (10.0^(-2.0))/6.582 # e/ħ prefactor, when multiplying Tesla gives 1/nm^2

magneticphase(ri, r₀, B::Real) =
@inbounds prefac * B/2 * (ri[1]*ri[2] - r₀[1]*ri[2] + ri[1]*r₀[2])
magneticphase(ri, r₀, B::Zero) = Zero()
