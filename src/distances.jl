using StaticArrays

"""
pbcdx(x1, x2, xsize)

Computes periodic boundary distance along one axis
"""
function pbcdx(x1, x2, xsize)
    dx = x2 - x1
    dx += -xsize * convert(Int32, round(dx/xsize))
    return(dx)
end

"""
pbcdistance(p1, p2, box)

Computes 3D periodic boundary distance between two points
"""
function pbcdistance(p1, p2, box)
    R2::Float32 = 0.
    @fastmath @inbounds for i in 1:3
        R2 += pbcdx(p1[i], p2[i], box[i])^2
    end
    R = sqrt(R2)
    return(R)
end

"""
buildDistanceMatrix(conf, box)

Builds the distance matrix for a given
configuration
"""
function builddistanceMatrix(conf, box)
    distanceMatrix = zeros(Float32, length(conf), length(conf))
    @inbounds for i in 1:length(conf)
        @inbounds for j in 1:length(conf)
            distanceMatrix[i,j] = pbcdistance(conf[i], conf[j], box)
        end
    end
    return(distanceMatrix)
end

"""
updatedistance(conf, box, distanceVector, pointIndex)

Updates all distances between a selected particle
and all the others in a given configuration
"""
function updatedistance!(conf, box, distanceVector, pointIndex)
    @fastmath @inbounds for i in 1:length(distanceVector)
        distanceVector[i] = pbcdistance(conf[i], conf[pointIndex], box)
    end
    return(distanceVector)
end
