module Graphene

using Interpolations
using StaticArrays
using Base.Iterators
using LinearAlgebra

SV = SVector{2, Float64}

const ccd = 0.142 # carbon carbon distance in nanometers
const a0 = √3 * ccd # in nanometers, lattice constant
const a1 = SV(a0, 0)  # bravais vector 1
const a2 = a0 .* SV(1/2, sqrt(3)/2) # bravais vector 2
const VALLEYS = # K₁, K₂, K₃, K₁', K₂', K₃'
@SVector [SV(4π/(3a0), 0), SV(-2π/(3a0), 2*√3*π/(3a0)), SV(-2π/(3a0), -2*√3*π/(3a0)),
 SV(-4π/(3a0), 0), SV(2π/(3a0), 2*√3*π/(3a0)), SV(2π/(3a0), -2*√3*π/(3a0)),
 SV(0, 0)]
const ħvF = 0.6582119514 #in units of eV*nm
const vF = 1e15 # in nm/sec
const K1 = SV(-2π/(3a0), 2*√3*π/(3a0))
const K2 = SV(2π/(3a0), 2*√3*π/(3a0))
const ħ = 6.582119514e-16  # in eV * seconds
const t = 2.8 # hoppping constant in eV

function dirac_sigma(E, Δk_over_k) #this σ gives **exact** position uncertainty
    k = abs(E/ħvF)
    # Divide by 2 due to uncertainty principle being ΔxΔk = 1/2 (for gaussian)
    σ = 1/(2Δk_over_k*k)
end
# E in eV, B in Tesla, result in nm
cyclotron_radius(E, B) = E*1000/B

##########################################
# energy dispersion and contour
##########################################
"""
    populate_contour(E, φs, ξ; kwargs...) → ks
Populate the energy contour of graphene at energy `E` at wavevectors oriented
at angles `φs` with respect to the center of the given valley `ξ`
(graphene has six valleys).

Notice that if `E > t` then the "valley" is always the center of the B.Z.
"""
function populate_contour(E, φs, xi; kwargs...)
    E ≥ t && (xi = 7)
    if abs(E) ≤ 1.0 # 2nd order expansion is accurate here
        kDiracs = wavevector_from_2ndorder.(E, φs; ξ = xi < 4 ? 1 : -1, kwargs...)
        k0s = [kD + VALLEYS[xi] for kD in kDiracs]
    else
        k0s = angle2wavevector.(φs, E; ξ = xi < 4 ? 1 : -1, kwargs...)
    end
    return k0s
end

dipsersion(k::AbstractArray; kwargs...) = dispersion(k[1], k[2]; kwargs...)
function dispersion(kx, ky; tprimeval = 0, λ=1, tval =t)
    a = ccd
    f = 2cos(√3*kx*a) + 4cos(√3*kx*a/2)*cos(3ky*a/2)
    ε = λ*tval*sqrt(3 + f) - tprimeval*f
end

function wavevector_from_2ndorder(E, φ; ξ=1, tprimeval = 0.0, tval = t)
    λ = sign(E)
    E = abs(E)
    α = (1/tval)*( (λ*tprimeval/tval) - ξ*cos(3φ)/6 )
    k = 2*(E/ħvF) / (1 + sqrt(1 + 4α*E))
    return SV(k*cos(φ), k*sin(φ))
end

"""
    groupvelocity(kx, ky; λ = 1, tprimeval = 0.0)
Return the group velocity at point `(kx, ky)` of momentum space.
"""
function groupvelocity(kx, ky; λ = 1, tprimeval = 0.0)
    # wavenumbers assumed to be in 1/nm
    t = t
    a = ccd
    s3 = √3
    f = 2cos(s3*kx*a) + 4cos(s3*kx*a/2)*cos(3*ky*a/2)
    sf3 = √(f + 3)
    vx = (s3*a/sf3) * (-λ*t + 2*tprimeval*sf3) *
         (   sin(s3*kx*a/2)*cos(3*ky*a/2) + sin(s3*kx*a)   )
    vy = (3a*cos(s3*kx*a/2)/ sf3) * (-λ*t + 2tprimeval*sf3) * sin(3*ky*a/2)
    return vx, vy
end
groupvelocity(k::Union{<:Array, <:Tuple}) = groupvelocity(k[1], k[2])

##########################################
# Convertions
##########################################
"""
    angle2wavevector(φ, E; ξ=1) → k
Create a wavevector `k` with angle `φ` at the energy contour `E`.

Choose the contour around the `ξ` valley (if E>t, thats the BZ center).
"""
function angle2wavevector(φ, E; ξ = 1, dE = 0.01, dk = 0.001)

    start = E > t ? 12.0 : 8.0
    E > t && (ξ = 7)
    c = VALLEYS[ξ]
    k = start
    ss, cs = sincos(φ)
    dir = SV(cs, ss)
    kx, ky = k*dir + c
    V = dispersion(kx, ky; λ = sign(E))
    while V ≥ E + dE
        k -= dk
        k ≤ 0 && error("Instability, k ≤ 0")
        kx, ky = k*dir + c
        V = dispersion(kx, ky; λ = sign(E))
    end
    return SV(kx, ky)
end

"""
    wavevector2angle(k0; kwargs...)
Transform a wavevector from an arbitrary location, to an angle.
Automatically decide the nearest valley to give a correct angle.
"""
function wavevector2angle(k0; kwargs...)
    E = dispersion(k0; kwargs...)
    kx, ky = k0
    if E < t
        xi = valley_index(kx, ky)
        kx, ky = bring_to_valley(kx, ky, xi)
        valley = xi ≤ 3 ? K1 : K2
        φ = round(atan(ky - valley[2], kx - valley[1]); digits = 8)
    else
        kx, ky = bring_to_center(kx, ky)
        φ = round(atan(ky, kx); digits = 8)
    end
    return φ
end

##########################################################################################
# Identify valleys / move from/to valleys
##########################################################################################
"""
    bring_to_valley(kx, ky) -> kx, ky
Bring wavevectors to a single valley (instead of the three equivalent).
The valley is the top one, either K_2 or K_2'.
"""
function bring_to_valley(kx, ky)
    xi = valley_index(kx, ky)
    a = ccd
    a1 = 4π/(3a) .* (-(√3) / 2, 1/2)
    a2 = 4π/(3a) .* (0.0, 1.0)
    b1 = 4π/(3a) .* ((√3) / 2, 1/2)
    if xi ≤ 3
        if kx > 0
            kx, ky = (kx, ky) .+ a1
        elseif ky < 0
            kx, ky = (kx, ky) .+ a2
        end
    else # valley K1', K2', K3'
        if kx < 0
            (kx, ky) = (kx, ky) .+ b1
        elseif ky < 0
            (kx, ky) = (kx, ky) .+ a2
        end
    end
    return (kx, ky)
end

valley_index(k0) = valley_index(k0[1], k0[2])
function valley_index(kx, ky) # 1 to 6
    φ = atan(ky, kx)
    if -π/6 ≤ φ < π/6;                   1
    elseif π/2 ≤ φ < π - π/6;            2
    elseif -π + π/6 ≤ φ < -π/2;          3
    elseif φ ≥ π - π/6 || φ < -π + π/6;  4
    elseif π/6 ≤ φ < π/2;                5
    elseif -π/2 ≤ φ < - π/6;             6
    end
end

function bring_to_center(kx, ky)
    a = 0.142
    a1 = 4π/(3a) .* ((√3) / 2, +1/2)
    a2 = 4π/(3a) .* (0.0, 1.0)

    if sqrt(kx^2 + ky^2) < 13.5
        return kx, ky
    end
    xsign = kx > 0 ? +1 : -1
    ysign = ky > 0 ? +1 : -1

    if abs(kx) < 12 # I need to transpose only up or down
        return kx, ky - a2[2]*ysign
    else
        return (kx, ky) .- (xsign, ysign) .* a1
    end
end

# This is useful when plotting:
function centralize(kx, ky, E)
    if abs(E) ≥ t
        kx, ky = bring_to_center(kx, ky)
    else
        kx, ky = bring_to_valley(kx, ky)
        z = 2*√3*π/(3a0)
        if ky < z
            ky += 2(z-ky)
        end
    end
    return kx, abs(ky)
end




end#module
