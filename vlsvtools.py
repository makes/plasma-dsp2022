import os
import sys
import glob
import logging
from collections.abc import Mapping
import numpy as np

from vdftools import VDF

PYTHON_INTERACTIVE_MODE = hasattr(sys, 'ps1')

SUPPRESS_ANALYSATOR_MESSAGES = True
if not PYTHON_INTERACTIVE_MODE or SUPPRESS_ANALYSATOR_MESSAGES:
    os.environ['PTNONINTERACTIVE'] = '1'  # suppress messages from analysator
os.environ['PTNOLATEX'] = '1'  # Latex rendering doesn't work with multithreading
sys.path.insert(0, 'analysator')
from analysator import pytools as pt

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# analysator.pyPlots.plot_vdf.verifyCellWithVspace does not filter out
# 0-size VDFs that occur inside earth so we need this to see if cell actually
# has VDF blocks.
def cell_has_vdf(handle, cellid, pop):
    cellswithblocks = np.atleast_1d(handle.read(mesh="SpatialGrid",
                                                tag="CELLSWITHBLOCKS",
                                                name=pop))
    blockspercell = np.atleast_1d(handle.read(mesh="SpatialGrid",
                                              tag="BLOCKSPERCELL",
                                              name=pop))
    emptycells = cellswithblocks[np.where(blockspercell == 0)[0]]
    return cellid in cellswithblocks and cellid not in emptycells

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
    def populations(self):
        return self.vlsvfile.handle.active_populations

    @property
    def vdf_cells(self):
        ret = [self] if self.has_vdf else []
        return ret

class SpatialMesh:
    def __init__(self, handle):
        self.shape = np.flip(handle.get_spatial_mesh_size())
        self.unsorted_cellids = handle.read_variable('cellid').astype(int)
        self.cellids = np.copy(self.unsorted_cellids)
        self.cellids.sort()

        pop = handle.active_populations[0]
        firstids = [cellid for cellid in self.cellids[:300] if cell_has_vdf(handle, cellid, pop)]
        assert len(firstids) >= 2
        self.vdf_spacing = firstids[1] - firstids[0]

    @property
    def cellid_matrix(self):
        return self.cellids.reshape(self.shape)

    @property
    def vdfcellids(self):
        s = self.vdf_spacing
        return self.cellid_matrix[::s,::s,::s].flatten()

class VLSVfile(Mapping):
    def __init__(self, filename):
        self.filename = filename
        self.handle = pt.vlsvfile.VlsvReader(file_name=filename)
        self.fileid = self.handle.read_parameter('fileindex')
        self.spatial_mesh = SpatialMesh(self.handle)
        logger.info(f'Found {len(self.cellids)} cells in {self.filename}')

        pop = self.populations[0]
        self.__vdfcellids = [cellid for cellid in self.spatial_mesh.vdfcellids
                             if cell_has_vdf(self.handle, cellid, pop)]

        logger.info(f'Found VDF data in {len(self.__vdfcellids)} cells')

        # preload all VDF cells
        self.__cells = {}
        for cellid in self.__vdfcellids:
            self.__cells[cellid] = VLSVcell(self, self.fileid, cellid, True)
        self.__mesh_size = self.handle.get_spatial_mesh_size()

    def __getitem__(self, key):
        if key in self.__cells.keys():
            return self.__cells[key]
        elif key in self.spatial_mesh.cellids:
            self.__cells[key] = VLSVcell(self, self.fileid, key, key in self.__vdfcellids)
        return self.__cells[key]

    def __len__(self):
        return len(self.spatial_mesh.cellids)

    def __iter__(self):
        for cellid in self.spatial_mesh.cellids:
            yield self.__getitem__(cellid)

    @property
    def cellids(self):
        return self.spatial_mesh.cellids

    @property
    def vdfcellids(self):
        return self.__vdfcellids

    @property
    def populations(self):
        return self.handle.active_populations

    @property
    def vdf_cells(self):
        ret = [cell for cell in self.__cells.values() if cell.has_vdf]
        return ret

    @property
    def mesh_size(self):
        return self.__mesh_size

    def get_vdf(self, cellid):
        return self[cellid].get_vdf()

    def has_vdf(self, cellid):
        return self[cellid].has_vdf

    def get_rho(self):
        cellids = self.handle.read_variable('cellid')
        ret = np.zeros([len(self), 4])
        for i, cellid in enumerate(sorted(cellids)):
            ret[i, :3] = self.handle.get_cell_coordinates(cellid)
        rho = self.handle.read_variable(f'{self.populations[0]}/vg_rho')
        ret[:, 3] = rho[cellids.argsort()]
        return ret

class VLSVfiles(Mapping):
    def __init__(self, filenames):
        self.filenames = filenames
        self.__files = {}
        for filename in filenames:
            file = VLSVfile(filename)
            self.__files[file.fileid] = file

    def __getitem__(self, key):
        return self.__files[key]

    def __len__(self):
        return len(self.__files)

    def __iter__(self):
        return self.__files.__iter__()

class VLSVdataset(Mapping):
    def __init__(self, path, filter='*.vlsv'):
        self.filenames = glob.glob(os.path.join(path, filter))
        logger.info(f'Found {len(self.filenames)} files.')
        self.__files = VLSVfiles(self.filenames)
        #self.__files = [VLSVfile(f) for f in self.filenames]
        #self.__files = sorted(self.__files, key=lambda f: f.fileid)
        #self.__cells = {}
        #for f in self.__files:
        #    for cell in f.values():
        #        self.__cells[(f.fileid, cell.cellid)] = cell

    def __getitem__(self, key):
        if len(key) == 1:
            return self.__files[key]
        elif len(key) == 2:
            print(key[0], key[1])
            return self.__files[key[0]][key[1]]
        else:
            raise KeyError("Invalid key")

    def __len__(self):
        return sum(len(f) for f in self.__files)

    def __iter__(self):
        for fileindex in range(len(self.__files)):
            for cellid in self.__files[fileindex].spatial_mesh.cellids:
                yield self.__files[fileindex].__getitem__(cellid)

    def get_rhos(self):
        return np.array([f.get_rho() for f in self.__files.values()])

    @property
    def files(self):
        return self.__files

    @property
    def fileids(self):
        return [f.fileid for f in self.__files.keys()]

    @property
    def vdf_cells(self):
        t = [f.vdf_cells for f in self.__files]
        return [cell for cells in t for cell in cells]
