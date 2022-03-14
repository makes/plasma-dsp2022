import sys
import numpy as np
sys.path.insert(0, 'analysator')
from analysator import pytools as pt

def findindex(i, vblocks, vblock_size, blocksize, vsize, sliceBz, sliceCz):
    iB = i // blocksize
    iBx = iB % vblocks[0]
    iBy = iB % sliceBz // vblocks[0]
    iBz = iB // sliceBz
    iCellInBlock = i % blocksize
    iCx = iCellInBlock % vblock_size[0]
    iCy = iCellInBlock % sliceCz // vblock_size[0]
    iCz = iCellInBlock // sliceCz
    iBCx = iBx*vblock_size[0] + iCx
    iBCy = iBy*vblock_size[1] + iCy
    iBCz = iBz*vblock_size[2] + iCz
    return int(iBCz*vsize[0]*vsize[1] + iBCy*vsize[0] + iBCx)

def reconstruct(vblock_size, vblocks, vcellids, vcellf):
    vblock_size = np.array(vblock_size)
    vblocks = np.array(vblocks)
    blocksize = np.prod(vblock_size)
    sliceBz = vblocks[0]*vblocks[1]
    vsize = vblock_size * vblocks
    sliceCz = vblock_size[0]*vblock_size[1]
    # Reconstruct the full velocity space
    VDF = np.zeros(np.prod(vsize))
    # Raw IDs are 0-based
    #for i in eachindex(vcellids)
    for i in range(len(vcellids)):
        iF = findindex(i, vblocks, vblock_size, blocksize, vsize, sliceBz, sliceCz)
        VDF[iF] = vcellf[i]
    return VDF.reshape(vsize)

if __name__ == "__main__":
    vlsv = pt.vlsvfile.VlsvReader(file_name='data/bulk.0000015.vlsv')
    velcells = vlsv.read_velocity_cells(1101, pop='proton')
    #vcellids = list(zip(*velcells.items()))
    vcellids = list(velcells.keys())
    vcellf = list(velcells.values())
    vblocks = vlsv.get_velocity_mesh_size(pop='proton')
    vblock_size = vlsv.get_velocity_block_size(pop="proton")
    print(len(vcellids), vcellids[0:3])
    print(len(vcellf), vcellf[0:3])
    print(vblocks)
    print(vblock_size)
    r = reconstruct(vblock_size, vblocks, vcellids, vcellf)
