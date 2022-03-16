import os
import sys
import glob
import numpy as np

PYTHON_INTERACTIVE_MODE = hasattr(sys, 'ps1')

SUPPRESS_ANALYSATOR_MESSAGES = True
if not PYTHON_INTERACTIVE_MODE or SUPPRESS_ANALYSATOR_MESSAGES:
    os.environ['PTNONINTERACTIVE'] = '1'  # suppress messages from analysator
os.environ['PTNOLATEX'] = '1'  # Latex rendering doesn't work with multithreading
sys.path.insert(0, 'analysator')
from analysator import pytools as pt
from analysator.pyPlots.plot_vdf import verifyCellWithVspace

class VDF:
    def __init__(self, cell):
        self.fileid = cell.fileid
        self.cellid = cell.cellid

        vlsv = cell.vlsvfile.handle

        velcells = vlsv.read_velocity_cells(self.cellid, pop='proton')
        vcellids = list(velcells.keys())
        vcellf = list(velcells.values())

        vblocks = vlsv.get_velocity_mesh_size(pop='proton')
        vblock_size = vlsv.get_velocity_block_size(pop="proton")
        self.data = VDF.__reconstruct(vblock_size, vblocks, vcellids, vcellf)

    @classmethod
    def __reconstruct(cls, vblock_size, vblocks, vcellids, vcellf):
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

    def transform_abs(self):
        self.data = np.abs(self.data)

    def transform_cbrt(self):
        self.data = np.cbrt(self.data)

    def transform_normalize(self):
        self.data = self.data / np.max(self.data)

class VLSVcell:
    def __init__(self, vlsvfile, fileid, cellid, has_vdf, x, y, z):
        self.vlsvfile = vlsvfile
        self.fileid = fileid
        self.cellid = cellid
        self.has_vdf = has_vdf
        self.x = x
        self.y = y
        self.z = z

    def get_vdf(self):
        return VDF(self)

    @property
    def vdf_cells_dict(self):
        ret = {self.cellid: self} if self.has_vdf else {}
        return ret

    @property
    def vdf_cells(self):
        return list(self.vdf_cells_dict.values())

class VLSVfile:
    def __init__(self, filename):
        self.filename = filename
        self.handle = pt.vlsvfile.VlsvReader(file_name=filename)
        self.fileid = self.handle.read_parameter('fileindex')
        self.cellids = sorted(self.handle.read_variable('cellid').astype(int).tolist())
        self.vdf_cellids = [id for id in self.cellids if verifyCellWithVspace(self.handle, id)]
        self.__cells = {}
        for cellid in self.cellids:
            has_vdf = True if cellid in self.vdf_cellids else False
            x, y, z = self.handle.get_cell_coordinates(cellid)
            self.__cells[cellid] = (VLSVcell(self, self.fileid, cellid, has_vdf, x, y, z))

    @property
    def cells_dict(self):
        return self.__cells

    @property
    def cells(self):
        return list(self.__cells.values())

    @property
    def vdf_cells_dict(self):
        return {cellid: self.__cells[cellid] for cellid in self.vdf_cellids}

    @property
    def vdf_cells(self):
        return list(self.vdf_cells_dict.values())

    def get_vdf(self, cellid):
        return self.__cells[cellid].get_vdf()

    def has_vdf(self, cellid):
        return self.__cells[cellid].has_vdf

class VLSVdataset:
    def __init__(self, path, filter='*.vlsv'):
        self.filenames = glob.glob(os.path.join(path, filter))
        self.files = [VLSVfile(f) for f in self.filenames]
        self.files = sorted(self.files, key=lambda f: f.fileid)

    @property
    def fileids(self):
        return [f.fileid for f in self.files]

    @property
    def cells_dict(self):
        c = {}
        for f in self.files:
            for cell in f.cells.values():
                c[(f.fileid, cell.cellid)] = cell
        return c

    @property
    def cells(self):
        return list(self.cells_dict.values())

    @property
    def vdf_cells_dict(self):
        c = {}
        for f in self.files:
            for cell in f.cells_dict.values():
                if cell.cellid in f.vdf_cellids:
                    c[(f.fileid, cell.cellid)] = cell
        return c

    @property
    def vdf_cells(self):
        return list(self.vdf_cells_dict.values())
