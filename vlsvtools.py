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

from juliacall import Main as jl
jl.seval("using Vlasiator")

class VDF:
    def __init__(self, data, fileid, cellid):
        self.fileid = fileid
        self.cellid = cellid
        self.data = data

    def find_peak(self):
        return np.unravel_index(np.argmax(self.data), self.data.shape)

    def transform_abs(self, inplace=False):
        if inplace:
            self.data = np.abs(self.data)
        else:
            return VDF(np.abs(self.data), self.fileid, self.cellid)

    def transform_cbrt(self, inplace=False):
        if inplace:
            self.data = np.cbrt(self.data)
        else:
            return VDF(np.cbrt(self.data), self.fileid, self.cellid)

    def transform_normalize(self, inplace=False):
        if inplace:
            self.data = self.data / np.max(self.data)
        else:
            return VDF(self.data / np.max(self.data), self.fileid, self.cellid)

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
        if not self.has_vdf:
            raise ValueError(f"Cell {self.cellid} does not contain VDF information.")
        jl_handle = self.vlsvfile.julia_handle
        vcellids, vcellf = jl.readvcells(jl_handle, self.cellid, species="proton")
        d = np.array(jl.Vlasiator.flatten(jl_handle.meshes["proton"],
                                             vcellids,
                                             vcellf), dtype=np.float32)
        return VDF(d, self.fileid, self.cellid)

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
        self.julia_handle = jl.load(self.filename)

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
