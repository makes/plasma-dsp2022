import os
import sys
from datetime import datetime
import logging

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
