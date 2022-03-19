import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

if not hasattr(sys, 'ps1'):  # if not in interactive mode
    mpl.use('agg')  # multithreading-compatible backend
    plt.ioff()

import vlsvtools
import vdftools
from misctools import output_subdir, filter_input_list, get_cpus, get_slurm_ids

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_projection(f, axis=1):
    return np.sum(f, axis=axis)

def get_slice(f, coords, axis=1):
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

def get_bounding_box(f):
    mask = f != 0
    bbox = tuple((np.min(indexes), np.max(indexes) + 1)
        for indexes in np.where(mask))
    return bbox


def plot_2d(f, ax=None, axis=1,
            pos=None, view='projection',
            crosshair=True, bbox=True, trim=False):
    if trim:
        f = trim_array(f, f != 0)

    if view == 'projection':
        imgdata = get_projection(f, axis=axis)
    elif view == 'slice':
        if pos is None:
            raise ValueError('pos must be specified for slice view')
        imgdata = get_slice(f, pos, axis=axis)
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
        bbox = get_bounding_box(f)
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
def plot_3d(f, ax=None, threshold=0.01, intensity=10.0):
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


def vdf_overview(vdf: vlsvtools.VDF):
    vdf.apply_transform(vdftools.TRANSFORM_ABS)
    vdf.apply_transform(vdftools.TRANSFORM_CBRT)

    peak = vdf.find_peak()

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    plot_2d(vdf.data, ax=ax[0][0], pos=peak, axis=0)
    plot_2d(vdf.data, ax=ax[0][1], pos=peak, axis=1)
    plot_2d(vdf.data, ax=ax[1][0], pos=peak, axis=2)
    ax[1][1].remove()
    ax[1][1] = fig.add_subplot(2,2,4,projection='3d')
    vdf.apply_transform(vdftools.TRANSFORM_NORMALIZE)
    plot_3d(vdf.data, ax=ax[1][1])
    fig.suptitle(f'file: {vdf.fileid:07} cell: {vdf.cellid:05}')
    fig.tight_layout()
    return fig

def plot(input, output_dir=None, plot_f=vdf_overview, dpi=300, jobid=None):
    if isinstance(input, vdftools.VDF):
        plot_f(input, output_dir)

    input_list = filter_input_list(input)

    if len(input_list) > 1:
        output_dir = output_subdir(output_dir, plot_f.__name__, jobid)

    if output_dir is None:  # not a batch run
        figs = []
        for cell in input_list:
            logger.info(f'loading cell {cell.fileid}:{cell.cellid}')
            vdf = cell.get_vdf()
            logger.info(f'processing VDF in cell {vdf.fileid}:{vdf.cellid}')
            fig = plot_f(vdf)
            figs.append(fig)
        #if len(figs) == 1:
        #    return figs[0]
        return figs

    # batch run: use multithreading

    num_cores = get_cpus()

    def build_output_filename(vdf, output_dir):
        if output_dir is None:
            return None
        filename = f'f{vdf.fileid:07}c{vdf.cellid:05}.png'
        return os.path.join(output_dir, filename)

    def process_vdf(vdf, filename):
        fig = plot_f(vdf)
        if filename is not None:
            fig.savefig(filename,
                        facecolor='white',
                        transparent=False,
                        dpi=dpi)
            plt.close(fig)

    USE_MULTITHREADING = True

    if not USE_MULTITHREADING:
        output_files = []
        for cell in input_list:
            logger.info(f'loading cell {cell.fileid}:{cell.cellid}')
            vdf = cell.get_vdf()
            logger.info(f'processing VDF in cell {vdf.fileid}:{vdf.cellid}')
            filename = build_output_filename(vdf, output_dir)
            process_vdf(vdf, filename)
            output_files.append(filename)

        return output_files

    def process_cell(cell):
        filename = build_output_filename(cell, output_dir)
        if os.path.exists(filename):
            logger.info(f'{filename} exists. Skipping.')
            return filename
        logger.info(f'loading cell {cell.fileid}:{cell.cellid}')
        vdf = cell.get_vdf()
        logger.info(f'processing VDF in cell {vdf.fileid}:{vdf.cellid}')
        process_vdf(vdf, filename)
        del vdf
        return filename

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        output_files = executor.map(process_cell, input_list)

    return list(output_files)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot VDF')
    parser.add_argument('inputdir', metavar='path', type=str, nargs='?', default='data', help='input directory')
    parser.add_argument('--outputdir', type=str, default='plots', help='output directory')
    #parser.add_argument('--dpi', type=int, default=300, help='dpi')
    #parser.add_argument('--jobid', type=int, help='jobid')
    args = parser.parse_args()

    jobid, array_task_id = get_slurm_ids()

    input = vlsvtools.VLSVdataset(args.inputdir)

    if array_task_id is not None:
        logger.debug(f'SLURM_ARRAY_TASK_ID={array_task_id}')
        input = input.files[array_task_id]

    input = input.vdf_cells

    plot(input, output_dir=args.outputdir, jobid=jobid)
