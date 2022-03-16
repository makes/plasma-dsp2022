import os
import logging
from datetime import datetime
import threading
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import vlsvtools

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
    vdf = vdf.transform_abs()
    vdf = vdf.transform_cbrt()

    peak = vdf.find_peak()

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    plot_2d(vdf.data, ax=ax[0][0], pos=peak, axis=0)
    plot_2d(vdf.data, ax=ax[0][1], pos=peak, axis=1)
    plot_2d(vdf.data, ax=ax[1][0], pos=peak, axis=2)
    ax[1][1].remove()
    ax[1][1] = fig.add_subplot(2,2,4,projection='3d')
    vdf = vdf.transform_normalize()
    plot_3d(vdf.data, ax=ax[1][1])
    fig.suptitle(f'file: {vdf.fileid:07} cell: {vdf.cellid:05}')
    fig.tight_layout()
    return fig

def plot(input, output_dir='plots', plot_f=vdf_overview, dpi=300, jobid=None):
    if isinstance(input, vlsvtools.VDF):
        plot_f(input, output_dir)

    input_list = None
    if isinstance(input, vlsvtools.VLSVcell):
        input_list = input.vdf_cells
    if isinstance(input, vlsvtools.VLSVfile):
        input_list = input.vdf_cells
    if isinstance(input, vlsvtools.VLSVdataset):
        input_list = input.vdf_cells
    if isinstance(input, list):
        input_list = list(filter(lambda x: x.has_vdf, input))
    if input_list is None:
        raise ValueError(f'Invalid input type {type(input)}. Supported types are VDF, VLSVcell, VLSVfile and VLSVdataset')

    if len(input_list) > 1 and output_dir is not None:
        date = datetime.today().strftime('%Y%m%d')
        def output_path(id_num, slurm):
            id_str = f'{id_num:04}' if not slurm else f'SLURM{id_num}'
            subdir = f'{plot_f.__name__}_{date}_{id_str}'
            return os.path.join(output_dir, subdir)
        if jobid is None:
            id_num = 1
            while os.path.exists(output_path(id_num, slurm=False)):
                id_num += 1
            output_dir = output_path(id_num, slurm=False)
            os.makedirs(output_dir)
        else:
            id_num = jobid
            output_dir = output_path(id_num, slurm=True)
            if not os.path.exists(output_dir):
                os.makedirs(output_path(id_num))

#    output_files = []
#    for i, cell in enumerate(input_list):
#        vdf = cell.get_vdf()
#        fig = plot_f(vdf)
#        if output_dir is not None:
#            filename = f'f{vdf.fileid:07}c{vdf.cellid:05}.png'
#            f = os.path.join(output_dir, filename)
#            fig.savefig(f,
#                        facecolor='white',
#                        transparent=False,
#                        dpi=dpi)
#            plt.close(fig)
#            output_files.append(f)
#    return output_files

    if output_dir is None:  # not a batch run
        figs = []
        for cell in input_list:
            logger.info(f'loading cell {cell.fileid}:{cell.cellid}')
            vdf = cell.get_vdf()
            print(f'processing VDF in cell {vdf.fileid}:{vdf.cellid}')
            fig = plot_f(vdf)
            figs.append(fig)
        if len(figs) == 1:
            return figs[0]
        return figs

    # batch run: use multithreading

    slurmcpus = os.getenv('SLURM_CPUS_PER_TASK')
    oscpus = os.cpu_count()
    if slurmcpus is not None:
        logger.debug(f"SLURM_CPUS_PER_TASK={slurmcpus} os.cpu_count()={oscpus}")
        num_cores = int(slurmcpus)
    else:
        num_cores = oscpus
    logger.info(f'{num_cores} cores detected')

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

    output_files = []
    vdfs = []
    for i, cell in enumerate(input_list):
        logger.info(f'loading cell {cell.fileid}:{cell.cellid}')
        vdf = cell.get_vdf()
        vdfs.append(vdf)
        if len(vdfs) == num_cores or i == len(input_list) - 1:
            threads = []
            for vdf in vdfs:
                logger.info(f'processing VDF in cell {vdf.fileid}:{vdf.cellid}')
                filename = build_output_filename(vdf, output_dir)
                output_files.append(filename)
                th = threading.Thread(target=process_vdf, args=[vdf, filename])
                th.start()
                threads.append(th)
            vdfs = []
            for th in threads:
                th.join()

    return output_files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot VDF')
    parser.add_argument('inputdir', metavar='path', type=str, nargs='?', default='data', help='input directory')
    parser.add_argument('--outputdir', type=str, default='plots', help='output directory')
    #parser.add_argument('--dpi', type=int, default=300, help='dpi')
    #parser.add_argument('--jobid', type=int, help='jobid')
    args = parser.parse_args()

    jobid = None
    array_task_id = None
    if os.getenv("SLURM_ARRAY_JOB_ID") != None:
        jobid = int(os.getenv("SLURM_ARRAY_JOB_ID"))
        array_task_id = int(os.getenv["SLURM_ARRAY_TASK_ID"])
    elif os.getenv("SLURM_JOB_ID") != None:
        jobid = int(os.getenv("SLURM_JOB_ID"))

    input = vlsvtools.VLSVdataset(args.inputdir)

    # TODO: parallelize by file for ARRAY_JOB
    if array_task_id is not None:
        logger.debug(f'SLURM_ARRAY_TASK_ID={array_task_id}')
        input = input.files[array_task_id]

    plot(input, output_dir=args.outputdir, jobid=jobid)
