import os
import sys
import numpy as np

sys.path.insert(0, 'analysator')
from analysator import pytools as pt

class VDFtransform:
    def __init__(self, transform_func, name):
        self.transform_func = transform_func
        self.name = name

class VDF:
    def __init__(self, cell):
        self.filename = cell.vlsvfile.filename
        self.fileid = cell.fileid
        self.cellid = cell.cellid
        self.transforms = []

        vlsv = pt.vlsvfile.VlsvReader(file_name=self.filename)

        velcells = vlsv.read_velocity_cells(self.cellid, pop=vlsv.active_populations[0])
        if not velcells:
            raise ValueError(f'No velocity data found for cell {self.fileid}:{self.cellid}')
        vcellids = list(velcells.keys())
        vcellf = list(velcells.values())

        vblocks = vlsv.get_velocity_mesh_size(pop=vlsv.active_populations[0])
        vblock_size = vlsv.get_velocity_block_size(pop=vlsv.active_populations[0])
        self.data = VDF.__reconstruct(vblock_size, vblocks, vcellids, vcellf)

    @classmethod
    def __reconstruct(cls, vblock_size, vblocks, vcellids, vcellf):
        # ported from https://github.com/henry2004y/Vlasiator.jl
        def findindex(i, vblocks, vblock_size, blocksize, vsize, sliceBz, sliceCz):
            iB = (i) // blocksize
            iBx = iB % vblocks[0]
            iBy = iB % sliceBz // vblocks[0]
            iBz = iB // sliceBz
            iCellInBlock = (i) % blocksize
            iCx = iCellInBlock % vblock_size[0]
            iCy = iCellInBlock % sliceCz // vblock_size[0]
            iCz = iCellInBlock // sliceCz
            iBCx = iBx*vblock_size[0] + iCx
            iBCy = iBy*vblock_size[1] + iCy
            iBCz = iBz*vblock_size[2] + iCz
            return iBCz*vsize[0]*vsize[1] + iBCy*vsize[0] + iBCx

        vblock_size = np.array(vblock_size)
        vblocks = np.array(vblocks)
        blocksize = np.prod(vblock_size)
        sliceBz = vblocks[0]*vblocks[1]
        vsize = vblock_size * vblocks
        sliceCz = vblock_size[0]*vblock_size[1]
        # Reconstruct the full velocity space
        VDF = np.zeros(np.prod(vsize))
        for i in range(len(vcellids)):
            j = findindex(vcellids[i], vblocks, vblock_size, blocksize, vsize, sliceBz, sliceCz)
            VDF[int(j)] = vcellf[i]
        return VDF.reshape(vsize).T

    def find_peak(self):
        return np.unravel_index(np.argmax(self.data), self.data.shape)

    def apply_transform(self, transform: VDFtransform):
        self.__apply_transform(transform.transform_func, transform.name)

    def __apply_transform(self, transform_func, name='unknown'):
        self.transforms.append(name)
        self.data = transform_func(self.data)

TRANSFORM_ABS = VDFtransform(np.abs, 'abs')
TRANSFORM_CBRT = VDFtransform(np.cbrt, 'cbrt')
TRANSFORM_NORMALIZE = VDFtransform(lambda x: x / np.max(x), 'normalize')
