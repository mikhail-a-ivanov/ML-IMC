"""
readXYZ(xyzname)

Reads an XYZ file and outputs coordinates of all atoms
arranged in separate frames
"""
function readXYZ(xyzname)
    file = open(xyzname, "r")
    lines = readlines(file)
    natoms = parse(Int, lines[1])
    ncomments = 2
    nlines = length(lines)
    nframes = Int((nlines % natoms) / 2)
    xyz = [[zeros(3) for i in 1:natoms] for j in 1:nframes]
    println("Reading $(xyzname) with $(nlines) lines...")
    println("Found $(nframes) frames with $(natoms) atoms each...")
    for lineId in 1:nlines
        frameId = ceil(Int, lineId/(natoms + ncomments))
        atomId = lineId - ((frameId - 1) * (natoms + ncomments)) - ncomments
        @assert atomId <= natoms
        if atomId > 0
            atomLine = split(lines[lineId])
            for i in 1:3
                xyz[frameId][atomId][i] = parse(Float64, atomLine[i+1])
            end
        end
    end
    close(file)
    println("Success! Closing the file...")
    return(xyz)
end

"""
readRDF(rdfname)

Reads RDF and distance histogram produced
by mcLJ.jl
"""
function readRDF(rdfname)
    file = open(rdfname, "r")
    lines = readlines(file)
    ncomments = 2
    nlines = length(lines) - ncomments
    bins = zeros(nlines)
    rdf = zeros(nlines)
    hist = zeros(nlines)
    for i in (1 + ncomments):length(lines)
        rdfline = split(lines[i])
        if length(rdfline) == 3
            bins[i - ncomments] = parse(Float64, rdfline[1])
            rdf[i - ncomments] = parse(Float64, rdfline[2])
            hist[i - ncomments] = parse(Float64, rdfline[3])
        end
    end
    return(bins, rdf, hist)
end
