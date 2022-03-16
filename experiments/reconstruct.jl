using Vlasiator

@inline function findindex(i, vblocks, vblock_size, blocksize, vsize, sliceBz, sliceCz)
    iB = (i - 1) ÷ blocksize
    iBx = iB % vblocks[1]
    iBy = iB % sliceBz ÷ vblocks[1]
    iBz = iB ÷ sliceBz
    iCellInBlock = (i - 1) % blocksize
    iCx = iCellInBlock % vblock_size[1]
    iCy = iCellInBlock % sliceCz ÷ vblock_size[1]
    iCz = iCellInBlock ÷ sliceCz
    iBCx = iBx*vblock_size[1] + iCx
    iBCy = iBy*vblock_size[2] + iCy
    iBCz = iBz*vblock_size[3] + iCz
    iOrigin = iBCz*vsize[1]*vsize[2] + iBCy*vsize[1] + iBCx + 1
 end

function reconstruct(vblock_size, vblocks, vcellids, vcellf)
    blocksize = prod(vblock_size)
    sliceBz = vblocks[1]*vblocks[2]
    vsize = @inbounds ntuple(i -> vblock_size[i] * vblocks[i], Val(3))
    sliceCz = vblock_size[1]*vblock_size[2]
    # Reconstruct the full velocity space
    VDF = zeros(Float32, vsize)
    # Raw IDs are 0-based
    @inbounds @simd for i in eachindex(vcellids)
       j = 1 + findindex(i, vblocks, vblock_size, blocksize, vsize, sliceBz, sliceCz)
       VDF[j] = vcellf[i]
    end

    VDF
 end

function flatten(vblock_size, vblocks, vcellids, vcellf)
    blocksize = prod(vblock_size)
    sliceBz = vblocks[1]*vblocks[2]
    vsizex, vsizey = vblock_size[1] * vblocks[1], vblock_size[2] * vblocks[2]
    sliceCz = vblock_size[1]*vblock_size[2]
    # Reconstruct the full velocity space
    VDFraw = zeros(Float32,
       vblock_size[1]*vblocks[1],
       vblock_size[2]*vblocks[2],
       vblock_size[3]*vblocks[3])
    # Fill nonzero values
    @inbounds @simd for i in eachindex(vcellids)
       VDFraw[vcellids[i]+1] = vcellf[i]
    end

    VDF = zeros(Float32,
       vblock_size[1]*vblocks[1],
       vblock_size[2]*vblocks[2],
       vblock_size[3]*vblocks[3])

    @inbounds @simd for i in eachindex(VDF)
       iB = (i - 1) ÷ blocksize
       iBx = iB % vblocks[1]
       iBy = iB % sliceBz ÷ vblocks[1]
       iBz = iB ÷ sliceBz
       iCellInBlock = (i - 1) % blocksize
       iCx = iCellInBlock % vblock_size[1]
       iCy = iCellInBlock % sliceCz ÷ vblock_size[1]
       iCz = iCellInBlock ÷ sliceCz
       iBCx = iBx*vblock_size[1] + iCx
       iBCy = iBy*vblock_size[2] + iCy
       iBCz = iBz*vblock_size[3] + iCz
       iF = iBCz*vsizex*vsizey + iBCy*vsizex + iBCx + 1
       VDF[iF] = VDFraw[i]
    end
    VDF
 end

file = "data/bulk.0000015.vlsv"
meta = load(file)

vcellids, vcellf = readvcells(meta, 1101, species="proton")
vblock_size = meta.meshes["proton"].vblock_size
vblocks = meta.meshes["proton"].vblocks

print(vblock_size)
print(vblocks)
print(flatten(vblock_size, vblocks, vcellids, vcellf)[1:10])