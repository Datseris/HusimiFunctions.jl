#=
Utility files for sorting the position vectors and optimizing which
indices of the wavefunction and space to use when doing
the Husimi computation.
=#
export xspan, sparse_sampling

function findclosest(val, A)
    i = 1
    d = abs(val - A[i])
    @inbounds for j in eachindex(A)
        dd = abs(val - A[j])
        if dd < d
            i = j
            d = dd
        end
    end
    return i
end


"""
    xslices(r) → i
Given `r` (a vector of `SVector`s), *which is assumed to be sorted*,
find the indices `i` that denote the changes in `x` slices.
"""
function xslices(r)
    xcol = [1]
    @inbounds for i in 2:length(r)
        r[i][1] == r[i-1][1] && continue
        push!(xcol, i)
    end
    return xcol
end

"""
    xspan(r0, r, toskip) → span
Find the `span` of `r` (the position vectors) that are within ± `toskip`
from the `x` location of `r0`, or within ± `toskip` from the maximum and
minimum `x` of `r0`, if `r0` is a vector of `SVector`s.
"""
function xspan(r0::SVector, ris, toskip)
    xmin = xmax = r0[1]
    xspan(xmin, xmax, ris, toskip)
end
function xspan(r0s::Vector{<:SVector}, ris, toskip)
    xmin, xmax = extrema(r[1] for r in r0s)
    xspan(xmin, xmax, ris, toskip)
end

function xspan(xmin, xmax, ris, toskip)
    xcol = xslices(ris)

    k = 0; i = 1 # left index, for minimum x
    @inbounds for (z, n) in enumerate(xcol)
        if ris[n][1] ≥ xmin - toskip
            i = n
            k = z
            break
        end
    end

    it = 0;  j=xcol[end] # right index, for maximum x
    @inbounds for n in @view xcol[k:end]
        it += 1
        if ris[n][1] ≥ xmax + toskip
            j = n
            break
        end
    end

    N = length(ris)
    # This line ensures that you get all the y elements
    # of the right-most column:
    j = (j==xcol[end] ? N : xcol[it+k+1]-1)
    return i:j
end

# TODO:
# Now we also limit the span to ys (assuming approximately same
# y span for every column)

"""
    sparse_sampling(ris; e = 2, ex = e, ey = e)
Return indices `i` that sample `ris` (the positions) by sparsely
sampling every `ex` x and every `ey` y positions (in integers).
"""
function sparse_sampling(ris; e = 2, ex = e, ey = e)
    xcol = xslices(ris)
    is = Int[]
    sizehint!(is, round(Int, length(ris)/(ex*ey)))
    for j in 1:ex:length(xcol)-1
        append!(is, xcol[j]:ey:xcol[j+1]-1)
    end
    return is
end
