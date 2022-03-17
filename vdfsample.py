import os
import logging
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import vlsvtools
import vdftools
from misctools import output_subdir, filter_input_list, get_cpus, get_slurm_ids

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VDFsample:
    def __init__(self, data):
        self.fileid = data['ids'][0]
        self.cellid = data['ids'][1]
        i = 0
        self.shells = []
        while f's{i}' in data:
            self.shells.append(data[f's{i}'])
            i += 1

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        return cls(data)


class VDFsampler:
    def __init__(self, n_spheres, n_points, r_max, distribution='radius'):
        self.n_spheres = n_spheres
        self.n_points = n_points
        self.r_max = r_max
        self.distribution = distribution
        self.spheres = self.__generate_spheres(n_spheres,
                                               n_points,
                                               r_max,
                                               distribution)
        self.transforms = []

    def __str__(self):
        ret = f'n_spheres: {self.n_spheres}\n'
        ret += f'n_points: {self.n_points}\n'
        ret += f'r_max: {self.r_max}\n'
        ret += f'distribution: {self.distribution}\n'
        ret += f'applied transformations: {self.transforms}\n'
        return ret

    def sample(self, input, transforms=[], output_dir=None, jobid=None):
        output_dir = output_subdir(output_dir, 'vdfsample', jobid)
        input_list = filter_input_list(input)

        self.transforms = [t.name for t in transforms]

        if len(input_list) <= 1:
            samples = []
            for cell in input_list:
                sample = self.__sample_cell(cell, transforms, output_dir)
                samples.append(sample)
            return samples
        else:  # batch job - multithread
            task_id = os.getenv("SLURM_ARRAY_TASK_ID")
            task_id = int(task_id) if task_id is not None else None
            if task_id == 0 or task_id is None:
                self.save(os.path.join(output_dir, 'VDFsampler.npz'))
            num_cores = get_cpus()
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                executor.map(lambda cell: self.__sample_cell(cell, transforms, output_dir), input_list)

    def save(self, filename):
        data = {}
        data['metadata'] = np.array([self.n_spheres, self.n_points, self.r_max, self.distribution])
        data['transforms'] = np.array(self.transforms)
        for i, sphere in enumerate(self.spheres):
            data[f's{i}'] = sphere
        np.savez(filename, **data)

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        n_spheres, n_points, r_max, distribution = data['metadata']
        n_spheres = int(n_spheres)
        n_points = int(n_points)
        r_max = float(r_max)
        new_smplr = cls(n_spheres, n_points, r_max, distribution)
        new_smplr.transforms = data['transforms'].tolist()
        i = 0
        spheres = []
        while f's{i}' in data:
            spheres.append(data[f's{i}'])
            i += 1
        new_smplr.spheres = spheres
        return new_smplr

    def __sample_cell(self, cell, transforms=[], output_dir=None):
        logger.info(f'sampling cell {cell.fileid}:{cell.cellid}')
        data = {}
        data['ids'] = np.array([cell.fileid, cell.cellid])
        vdf = cell.get_vdf()
        for t in transforms:
            vdf.apply_transform(t)
        peak = vdf.find_peak()
        for i, sphere in enumerate(self.spheres):
            s = sphere + peak
            data[f's{i}'] = self.__sample_sphere(vdf.data, s)
        if output_dir is not None:
            filename = f'f{cell.fileid:07}c{cell.cellid:05}.npz'
            np.savez(os.path.join(output_dir, filename), **data)
        return VDFsample(data)

    # https://stackoverflow.com/a/26127012/287954
    @classmethod
    def fibonacci_sphere(cls, samples=1000):
        points = np.empty((samples, 3))
        phi = np.pi * (3 - 5 ** 0.5)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = (1 - y * y) ** 0.5  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points[i] = np.array([x, y, z])

        return points

    def __generate_spheres(self, n_spheres, n_points, r_max, distribution):
        radii = np.linspace(r_max / n_spheres, r_max, n_spheres)
        if distribution == 'area':
            areas = 4 * np.pi * radii ** 2
            a_tot = np.sum(areas)
            points_per_sphere = np.trunc((areas / a_tot) * n_points).astype(int)
            points_per_sphere[:n_points - np.sum(points_per_sphere)] += 1
        elif distribution == 'radius':
            points_per_sphere = np.trunc((radii / np.sum(radii)) * n_points).astype(int)
            points_per_sphere[:n_points - np.sum(points_per_sphere)] += 1
        elif distribution == 'even':
            points_per_sphere = np.ones(n_spheres) * (n_points // n_spheres)
            points_per_sphere[-(n_points - np.sum(points_per_sphere)):] += 1
        else:
            raise ValueError(f'Unknown distribution {distribution}')
        spheres = []
        for n, radius in zip(points_per_sphere, radii):
            s = np.array(radius * VDFsampler.fibonacci_sphere(n))
            spheres.append(np.trunc(s).astype(int))
        return spheres

    def __sample_sphere(self, f, sphere):
        return f[sphere[:, 0], sphere[:, 1], sphere[:, 2]]

def load_samples(path):
    samplerfile = os.path.join(path, 'VDFsampler.npz')
    if not os.path.exists(samplerfile):
        logger.warn(f'VDFsampler.npz not found in {path}')
        sampler = None
    else:
        sampler = VDFsampler.from_file(samplerfile)
    samplefiles = glob.glob(os.path.join(path, 'f*.npz'))
    samples = [VDFsample.from_file(f) for f in samplefiles]
    samples = sorted(samples, key=lambda s: (s.fileid, s.cellid))
    return sampler, samples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sample VDFs')
    parser.add_argument('inputdir', metavar='path', type=str, nargs='?', default='data', help='input directory')
    parser.add_argument('--outputdir', type=str, default='samples', help='output directory')
    parser.add_argument('-s', '--spheres', help='Number of spheres', default=25, type=int)
    parser.add_argument('-p', '--points', help='Total number of points', default=100000, type=int)
    parser.add_argument('-r', '--rmax', help='Sampling radius', default=90, type=float)
    parser.add_argument('-d', '--distribution', help='Distribution of points', default='radius', type=str)
    args = parser.parse_args()

    transforms = [vdftools.TRANSFORM_ABS, vdftools.TRANSFORM_CBRT]
    sampler = VDFsampler(args.spheres,
                         args.points,
                         args.rmax,
                         args.distribution)

    input = vlsvtools.VLSVdataset(args.inputdir)

    jobid, array_task_id = get_slurm_ids()

    if array_task_id is not None:
        logger.debug(f'SLURM_ARRAY_TASK_ID={array_task_id}')
        input = input.files[array_task_id]

    input = input.vdf_cells

    sampler.sample(input, transforms=transforms, output_dir=args.outputdir, jobid=jobid)
