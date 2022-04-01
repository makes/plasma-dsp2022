import os
import sys
from datetime import datetime
import logging
import numpy as np

sys.path.insert(0, 'analysator')
from analysator.pyCalculations.rotation import rotation_array_matrix
import scipy.spatial.transform

import vlsvtools

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_cpus():
    slurmcpus = os.getenv('SLURM_CPUS_PER_TASK')
    oscpus = os.cpu_count()
    if slurmcpus is not None:
        logger.debug(f"SLURM_CPUS_PER_TASK={slurmcpus} os.cpu_count()={oscpus}")
        num_cores = int(slurmcpus)
    else:
        num_cores = oscpus
    logger.info(f'{num_cores} cores detected')
    return num_cores

def get_slurm_ids():
    jobid = None
    array_task_id = None
    if os.getenv("SLURM_ARRAY_JOB_ID") != None:
        jobid = int(os.getenv("SLURM_ARRAY_JOB_ID"))
        array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    elif os.getenv("SLURM_JOB_ID") != None:
        jobid = int(os.getenv("SLURM_JOB_ID"))
    return jobid, array_task_id

def output_subdir(basedir, basename, slurm_jobid=None):
    if basedir is None:
        return None
    date = datetime.today().strftime('%Y%m%d')
    def output_path(id_num, slurm):
        id_str = f'{id_num:04}' if not slurm else f'SLURM{id_num}'
        subdir = f'{basename}_{date}_{id_str}'
        return os.path.join(basedir, subdir)
    if slurm_jobid is None:
        id_num = 1
        while os.path.exists(output_path(id_num, slurm=False)):
            id_num += 1
        output_dir = output_path(id_num, slurm=False)
        os.makedirs(output_dir)
    else:
        id_num = slurm_jobid
        output_dir = output_path(id_num, slurm=True)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except FileExistsError:
                pass
    return output_dir

def filter_input_list(input):
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
    return input_list

def rotateArrayVectorToVector(vector1, vector2, axis=2):
    ''' Applies rotation matrix that would rotate vector2 to z-axis on vector1 and then returns the rotated vector1. This is the fuction rotateVectorToVector from https://github.com/fmihpc/analysator/blob/master/pyCalculations/rotation.py modified to support NumPy vectorization.
        :param vector1        Vector to be rotated
        :param vector2        Vector for creating the rotation matrix
        :returns rotated vector1 vector
        .. note::
            vector1 and vector2 must be 3d vectors
    '''
    basis = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])][axis]
    vector_u = np.cross(vector2, basis)
    with np.errstate(divide='ignore', invalid='ignore'):
        vector_u = vector_u / np.linalg.norm(vector_u, axis=1).reshape((-1,1))
    angle = np.arccos(vector2.dot(basis) / np.linalg.norm(vector2, axis=1))
    nulls = np.argwhere(np.isnan(angle))[:,0]
    angle[nulls] = 0
    # A unit vector version of the given vector
    R = rotation_array_matrix(vector_u, angle)
    rot = scipy.spatial.transform.Rotation.from_matrix(R)
    # Rotate vector
    rotated = rot.apply(vector1)
    nulls = np.argwhere(np.isnan(rotated))[:,0]
    rotated[nulls] = vector1[nulls]  # if norm == 0: return vector1
    return rotated