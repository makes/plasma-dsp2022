import os
import sys
import glob
import threading
import numpy as np
import pandas as pd
import scipy.stats as scs
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, 'analysator')
from analysator import pytools as pt
from analysator.pyPlots.plot_vdf import verifyCellWithVspace

from juliacall import Main as jl

def transform_f(data):
    data = np.abs(data)
    data = np.cbrt(data) # square root transform to enhance the more distant parts
    data = data / np.max(data)
    return data

def find_peak_f_coordinates(data):
    return np.unravel_index(np.argmax(data), data.shape)

def get_projection(f, axis=1):
    return np.sum(f, axis=axis)

def slice_cutout(f, coords, axis=1):
    if axis == 0:
        return f[coords[0], :, :]
    elif axis == 1:
        return f[:, coords[1], :]
    elif axis == 2:
        return f[:, :, coords[2]]
    else:
        raise ValueError(f'Unknown axis {axis}')

# https://stackoverflow.com/a/54571830/287954
def trim_array(arr, mask):
    bounding_box = tuple(
        slice(np.min(indexes), np.max(indexes) + 1)
        for indexes in np.where(mask))
    return arr[bounding_box]

def get_bbox(arr, mask):
    bounding_box = tuple((np.min(indexes), np.max(indexes) + 1)
        for indexes in np.where(mask))
    return bounding_box

def plot_f_2d(f, ax=None, axis=1,
              pos=None, view='projection',
              crosshair=True, bbox=True, trim=False):
    if trim:
        f = trim_array(f, f != 0)

    if view == 'projection':
        imgdata = get_projection(f, axis=axis)
    elif view == 'slice':
        if pos is None:
            raise ValueError('pos must be specified for slice view')
        imgdata = slice_cutout(f, pos, axis=axis)
    else:
        raise ValueError(f'Unknown view {view}')

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(imgdata, cmap='nipy_spectral_r', interpolation='none')

    if crosshair:
        if pos is None:
            raise ValueError('pos must be specified for crosshair')
        # axis: 0=zy, 1=xz, 2=xy
        xy = [pos[i] for i in range(len(pos)) if i != axis]
        ax.axvline(x=xy[1], color='b', linestyle='-', linewidth=0.5)
        ax.axhline(y=xy[0], color='b', linestyle='-', linewidth=0.5)
        circ = mpl.patches.Circle((xy[1], xy[0]), 90,
                                  linewidth=0.5,
                                  edgecolor='b',
                                  facecolor='none')
        ax.add_patch(circ)

    if bbox and not trim:
        bbox = get_bbox(f, f != 0)
        bbox = [bbox[i] for i in range(len(bbox)) if i != axis]
        a = (bbox[1][0], bbox[0][0]) # rectangle anchor point
        w = bbox[1][1] - bbox[1][0] # rectangle width
        h = bbox[0][1] - bbox[0][0] # rectangle height
        rect = mpl.patches.Rectangle(a, w, h,
                                     linewidth=0.5,
                                     edgecolor='k',
                                     linestyle='--',
                                     facecolor='none')
        ax.add_patch(rect)

# https://stackoverflow.com/a/66939879/287954
def plot_f_3d(f, ax=None, threshold=0.01, intensity=10.0, trim=True):
    f = trim_array(f, f != 0)
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = np.abs(f) > threshold
    idx = np.arange(int(np.prod(f.shape)))
    x, y, z = np.unravel_index(idx, f.shape)
    ax.scatter(x, y, z, c=f.flatten(), s=intensity * mask, edgecolor="face", alpha=0.2, marker="o", cmap="nipy_spectral", linewidth=0)
    ax.set(xlabel="x", ylabel="y", zlabel="z")

def plot_vdf(f, idx, cellid):
    peak = find_peak_f_coordinates(f)

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    plot_f_2d(f, ax=ax[0][0], pos=peak, axis=0)
    plot_f_2d(f, ax=ax[0][1], pos=peak, axis=1)
    plot_f_2d(f, ax=ax[1][0], pos=peak, axis=2)
    ax[1][1].remove()
    ax[1][1] = fig.add_subplot(2,2,4,projection='3d')
    plot_f_3d(f, ax=ax[1][1])
    plt.tight_layout()
    plt.savefig(f'plots/f{idx}c{cellid}.png')

def load_f(meta, cell):
    vcellids, vcellf = jl.readvcells(meta, cell, species="proton")
    return np.array(jl.Vlasiator.flatten(meta.meshes["proton"], vcellids, vcellf), dtype=np.float64)

def process_f(f, fileIndex, cell):
    f = transform_f(f)
    plot_vdf(f, fileIndex, cell)

def process_file(filename, num_cores):
    print(f'Processing {filename}')
    vlsv = pt.vlsvfile.VlsvReader(file_name=filename)
    idx = vlsv.read_parameter('fileIndex')
    cell_ids = vlsv.read_variable("CellID")
    cells_with_vdf = [int(id) for id in cell_ids if verifyCellWithVspace(vlsv, id)]
    print(cells_with_vdf)
    meta = jl.load(filename)
    process_f_params = []
    for cell in cells_with_vdf:
        print(f'loading cell {cell}')
        f = load_f(meta, cell)
        process_f_params.append((f, idx, cell))
        if len(process_f_params) == num_cores or cell == cells_with_vdf[-1]:
            threads = []
            for params in process_f_params:
                print(f'processing cell {params[2]}')
                th = threading.Thread(target=process_f, args=params)
                th.start()
                threads.append(th)
            process_f_params = []
            for th in threads:
                th.join()

    print(f'Finished processing {filename}')

if __name__ == "__main__":
    slurmcpus = os.getenv('SLURM_CPUS_PER_TASK')
    file_idx = None
    if slurmcpus is not None:
        print("Running on Slurm")
        num_cores = int(slurmcpus)
        file_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        print(f'Slurm task id: {file_idx}')
    else:
        num_cores = os.cpu_count()

    print(f'{num_cores} cores detected')
    os.environ['PTNONINTERACTIVE'] = '1'  # suppress messages from analysator
    jl.seval("using Vlasiator")

    mpl.use('agg')  # use non-qt backend for multithreading

    if file_idx is None:
        data_filenames = glob.glob('data/*.vlsv')
    else:
        data_filenames = [f'data/bulk.{file_idx:07}.vlsv']

    for filename in data_filenames:
        process_file(filename, num_cores)
