using Test, HusimiFunctions

r0 = SVector(0.5, 0.5)
k0 = SVector(0.5, 0.25)

using Random
Random.seed!(5)
r = [rand(SVector{2}) for i in 1:10000]
ψ = [rand(ComplexF64) for i in 1:10000]

Q1 = husimi(ψ, r, r0, k0, 2.0)
@test Q1/1e6 ≈ 1.882235 atol=1e-6
Q2 = husimi(ψ, r, r0, k0, 2.0, 5.0)
@test Q2/1e6 ≈ 1.881620 atol = 1e-6

# # non magnetic
# @btime husimi(ψ, r, r0, k0, 2.0; thread = false)
# @btime husimi(ψ, r, r0, k0, 2.0; thread = true)
#
# # non magnetic
# @btime husimi(ψ, r, r0, k0, 2.0, 0.5; thread = false)
# @btime husimi(ψ, r, r0, k0, 2.0, 0.5; thread = true)
