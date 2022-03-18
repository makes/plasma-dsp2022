import os
import sys
import glob
import logging
from collections.abc import Mapping
import numpy as np

from vdftools import VDF, POPULATION

PYTHON_INTERACTIVE_MODE = hasattr(sys, 'ps1')

SUPPRESS_ANALYSATOR_MESSAGES = True
if not PYTHON_INTERACTIVE_MODE or SUPPRESS_ANALYSATOR_MESSAGES:
    os.environ['PTNONINTERACTIVE'] = '1'  # suppress messages from analysator
os.environ['PTNOLATEX'] = '1'  # Latex rendering doesn't work with multithreading
sys.path.insert(0, 'analysator')
from analysator import pytools as pt
from analysator.pyPlots.plot_vdf import verifyCellWithVspace

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VLSVcell(Mapping):
    def __init__(self, vlsvfile, fileid, cellid, has_vdf):
        self.vlsvfile = vlsvfile
        self.fileid = fileid
        self.cellid = cellid
        self.has_vdf = has_vdf
        self.__spatial_coords = None

    def __getitem__(self, key):
        return {self.cellid: self}[key]

    def __len__(self):
        return 1

    def __iter__(self):
        return iter({self.cellid: self})

    def get_vdf(self):
        return VDF(self)

    @property
    def coordinates(self):
        if self.__spatial_coords is None:
            self.__spatial_coords = self.vlsvfile.handle.get_cell_coordinates(self.cellid)
        return self.__spatial_coords

    @property
    def vdf_cells(self):
        ret = [self] if self.has_vdf else []
        return ret

class VLSVfile(Mapping):
    def __init__(self, filename):
        self.filename = filename
        self.handle = pt.vlsvfile.VlsvReader(file_name=filename)
        self.fileid = self.handle.read_parameter('fileindex')
        self.cellids = sorted(self.handle.read_variable('cellid').astype(int).tolist())
        logger.info(f'Found {len(self.cellids)} cells in {self.filename}')

        def find_vdfcellids(handle, cellids):
            hasvdf = verifyCellWithVspace
            firstids = [cellid for cellid in cellids[:300] if hasvdf(handle, cellid)]
            diff = firstids[1] - firstids[0]
            return list(range(firstids[0], diff, len(cellids)))
        self.vdfcellids = find_vdfcellids(self.handle, self.cellids)

        self.__cells = {}
        for cellid in self.cellids:
            self.__cells[cellid] = VLSVcell(self, self.fileid, cellid, cellid in self.vdfcellids)

    def __getitem__(self, key):
        return self.__cells[key]

    def __len__(self):
        return len(self.__cells)

    def __iter__(self):
        return iter(self.__cells)

    @property
    def vdf_cells(self):
        return [cell for cell in self.values() if cell.has_vdf]

    def get_vdf(self, cellid):
        return self.__cells[cellid].get_vdf()

    def has_vdf(self, cellid):
        return self.__cells[cellid].has_vdf

    def get_rho(self):
        cellids = self.handle.read_variable('cellid')
        ret = np.zeros([len(self), 4])
        for i, cellid in enumerate(sorted(cellids)):
            ret[i, :3] = self.handle.get_cell_coordinates(cellid)
        rho = self.handle.read_variable(f'{POPULATION}/vg_rho')
        ret[:, 3] = rho[cellids.argsort()]
        return ret

class VLSVdataset(Mapping):
    def __init__(self, path, filter='*.vlsv'):
        self.filenames = glob.glob(os.path.join(path, filter))
        logger.info(f'Found {len(self.filenames)} files.')
        self.__files = [VLSVfile(f) for f in self.filenames]
        self.__files = sorted(self.__files, key=lambda f: f.fileid)
        self.__cells = {}
        for f in self.__files:
            for cell in f.values():
                self.__cells[(f.fileid, cell.cellid)] = cell

    def __getitem__(self, key):
        return self.__cells[key]

    def __len__(self):
        return len(self.__cells)

    def __iter__(self):
        return iter(self.__cells)

    def get_rhos(self):
        return np.array([f.get_rho() for f in self.__files])

    @property
    def files(self):
        return self.__files

    @property
    def fileids(self):
        return [f.fileid for f in self.__files]

    @property
    def vdf_cells_dict(self):
        c = {}
        for f in self.__files:
            for cell in f.cells_dict.values():
                if cell.cellid in f.vdf_cellids:
                    c[(f.fileid, cell.cellid)] = cell
        return c

    @property
    def vdf_cells(self):
        return [cell for cell in self.values() if cell.has_vdf]
