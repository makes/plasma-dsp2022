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

        self._B = None
        self._E = None
        self._rho = None
        self._V = None
        self._P = None

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

    def has_vlsv_var(self, varname):
        return varname in self.handle.get_all_variables()

    def read_vlsv_var(self, varname):
        assert self.has_vlsv_var(varname)
        return self.handle.read_variable(varname)

    @property
    def B(self):
        if self._B is not None:
            return self._B

        if self.has_vlsv_var('B'):
            self._B = self.read_vlsv_var('B')
        elif self.has_vlsv_var('fg_b'):
            self._B = self.read_vlsv_var('fg_b')

        self._B = self._B[self.spatial_mesh.unsorted_cellids.argsort()]
        return self._B

    @property
    def E(self):
        if self._E is not None:
            return self._E

        if self.has_vlsv_var('E'):
            self._E = self.read_vlsv_var('E')
        elif self.has_var('fg_e'):
            self._E = self.read_vlsv_var('fg_e')

        self._E = self._E[self.spatial_mesh.unsorted_cellids.argsort()]
        return self._E

    @property
    def rho(self):
        if self._rho is not None:
            return self._rho

        if self.has_vlsv_var('rho'):
            self._rho = self.read_vlsv_var('rho')
        elif self.has_var(f'{self.populations[0]}/vg_rho'):
            self._rho = self.read_vlsv_var(f'{self.populations[0]}/vg_rho')

        self._rho = self._rho[self.spatial_mesh.unsorted_cellids.argsort()]
        return self._rho

    @property
    def V(self):
        if self._V is not None:
            return self._V

        if self.has_vlsv_var('rho_v'):
            with np.errstate(divide='ignore', invalid='ignore'):
                self._V = self.read_vlsv_var('rho_v') / self.read_vlsv_var('rho').reshape((-1,1))
            self._V[np.isnan(self._V)] = 0
        elif self.has_vlsv_var(f'{self.populations[0]}/vg_v'):
            self._V = self.read_vlsv_var(f'{self.populations[0]}/vg_v')

        self._V = self._V[self.spatial_mesh.unsorted_cellids.argsort()]
        return self._V

    @property
    def P(self):
        if self._P is not None:
            return self._P

        if self.has_vlsv_var('PTensorDiagonal'):
            p_diag = self.read_vlsv_var('PTensorDiagonal')
        elif self.has_vlsv_var(f'{self.populations[0]}/vg_ptensor_diagonal'):
            p_diag = self.read_vlsv_var(f'{self.populations[0]}/vg_ptensor_diagonal')

        if self.has_vlsv_var('PTensorOffDiagonal'):
            p_offdiag = self.read_vlsv_var('PTensorOffDiagonal')
        elif self.has_vlsv_var(f'{self.populations[0]}/vg_ptensor_offdiagonal'):
            p_offdiag = self.read_vlsv_var(f'{self.populations[0]}/vg_ptensor_offdiagonal')

        N = p_diag.shape[1]
        P = np.expand_dims(p_diag, axis=1)
        P = P*np.eye(N)
        P[:,0,1] = p_offdiag[:,2]
        P[:,0,2] = p_offdiag[:,1]
        P[:,1,2] = p_offdiag[:,0]
        P[:,1,0] = P[:,0,1]
        P[:,2,0] = P[:,0,2]
        P[:,2,1] = P[:,1,2]

        self._P = P[self.spatial_mesh.unsorted_cellids.argsort()]
        return self._P

    def get_rho(self):
        cellids = self.read_var('cellid')
        ret = np.zeros([len(self), 4])
        for i, cellid in enumerate(sorted(cellids)):
            ret[i, :3] = self.handle.get_cell_coordinates(cellid)
        rho = self.read_var(f'{self.populations[0]}/vg_rho')
        ret[:, 3] = rho[cellids.argsort()]
        return ret

    def get_vdf(self, cellid):
        return self[cellid].get_vdf()

    def has_vdf(self, cellid):
        return self[cellid].has_vdf


class VLSVfiles(Mapping):
    def __init__(self, filenames):
        self.filenames = filenames
        files = [VLSVfile(filename) for filename in filenames]
        files = sorted(files, key=lambda f: f.fileid)
        self.__files = {f.fileid: f for f in files}

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

    def __getitem__(self, key):
        if len(key) == 1:
            return self.__files[key]
        elif len(key) == 2:

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
        return list(self.__files.values())

    @property
    def fileids(self):
        return [f.fileid for f in self.__files.keys()]

    @property
    def vdf_cells(self):
        t = [f.vdf_cells for f in self.__files.values()]
        return [cell for cells in t for cell in cells]
