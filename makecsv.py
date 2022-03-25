import os
import logging
import numpy as np
import scipy.stats as scs
import pandas as pd

import vlsvtools
import vdfsample


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SAMPLEPATH = 'samples/2D_magnetosphere'
#SAMPLEPATH = 'samples/vdfsample_20220317_SLURM71721393'
#IMGPATH = 'plots/vdf_overview_20220317_SLURM71721410'
DATAPATH = '2D_magnetosphere'
#DATAPATH = 'data'
CSVFILE = 'samples/2D_magnetosphere/2D_magnetosphere.csv'
#CSVFILE = 'output/vdfsample_20220317_SLURM71721393.csv'

def generate_feature_zeros(row):
    zeros = 0
    n = 0
    while f's{n}_zeros' in row:
        zeros += 1 if row[f's{n}_zeros'] == 1 else 0
        n += 1
    return zeros / n

def generate_feature_partials(row):
    partials = 0
    n = 0
    while f's{n}_zeros' in row:
        if row[f's{n}_zeros'] > 0 and row[f's{n}_zeros'] < 1:
            partials += 1
        n += 1
    return partials / n

def generate_feature_mean(row):
    """ Mean of means """
    means = []
    n = 0
    while f's{n}_mean' in row:
        means.append(row[f's{n}_mean'])
        n += 1
    return np.mean(means)

def generate_feature_gap(row):
    gap = None
    n = 0
    while f's{n}_zeros' in row:
        if row[f's{n}_zeros'] == 1 and gap is None:
            gap = 0
        elif row[f's{n}_zeros'] != 1 and gap == 0:
            gap = 1
        n += 1
    if gap is None:
        gap = 0
    return gap

def collect_dataframe(samples, vlsvdata):
    df = None
    for sample in samples:
        logger.info(f"processing sample {sample.fileid}:{sample.cellid}")
        ids = {'fileid': sample.fileid, 'cellid': sample.cellid}
        cols = {}
        for i, s in enumerate(sample.shells):
            cols[f's{i}_min'] = np.min(s)
            cols[f's{i}_max'] = np.max(s)
            cols[f's{i}_mean'] = np.mean(s)
            cols[f's{i}_median'] = np.median(s)
            cols[f's{i}_var'] = np.var(s)
            cols[f's{i}_skew'] = scs.skew(s)
            cols[f's{i}_kurt'] = scs.kurtosis(s)
            cols[f's{i}_zeros'] = np.sum(s == 0) / len(s)
        row = pd.DataFrame([ids | cols])
        row['mean'] = generate_feature_mean(cols)
        row['zeros'] = generate_feature_zeros(cols)
        row['partials'] = generate_feature_partials(cols)
        row['gap'] = generate_feature_gap(cols)

        cell = vlsvdata[(sample.fileid, sample.cellid)]
        x, y, z = cell.coordinates
        row['spatial_x'], row['spatial_y'], row['spatial_z'] = x, y, z

        if 'rho' in cell.vlsvfile.handle.get_all_variables():

            m0_density = cell.vlsvfile.handle.read_variable('rho', sample.cellid)
        elif f'{cell.populations[0]}/vg_rho' in cell.vlsvfile.handle.get_all_variables():
            m0_density = cell.vlsvfile.handle.read_variable(f'{cell.populations[0]}/vg_rho', sample.cellid)
        row['m0_density'] = m0_density

        if 'rho_v' in cell.vlsvfile.handle.get_all_variables():
            m1_velocity = cell.vlsvfile.handle.read_variable('rho_v', sample.cellid) / m0_density
        elif f'{cell.populations[0]}/vg_v' in cell.vlsvfile.handle.get_all_variables():
            m1_velocity = cell.vlsvfile.handle.read_variable(f'{cell.populations[0]}/vg_v', sample.cellid)
        row['m1_velocity_0'] = m1_velocity[0]
        row['m1_velocity_1'] = m1_velocity[1]
        row['m1_velocity_2'] = m1_velocity[2]
        row['m1_speed'] = np.linalg.norm(m1_velocity)

        if 'PTensorDiagonal' in cell.vlsvfile.handle.get_all_variables():
            m2_pressure_diag = cell.vlsvfile.handle.read_variable('PTensorDiagonal', sample.cellid)
        elif f'{cell.populations[0]}/vg_ptensor_diagonal' in cell.vlsvfile.handle.get_all_variables():
            m2_pressure_diag = cell.vlsvfile.handle.read_variable(f'{cell.populations[0]}/vg_ptensor_diagonal', sample.cellid)

        if 'PTensorOffDiagonal' in cell.vlsvfile.handle.get_all_variables():
            m2_pressure_offdiag = cell.vlsvfile.handle.read_variable('PTensorOffDiagonal', sample.cellid)
        elif f'{cell.populations[0]}/vg_ptensor_offdiagonal' in cell.vlsvfile.handle.get_all_variables():
            m2_pressure_offdiag = cell.vlsvfile.handle.read_variable(f'{cell.populations[0]}/vg_ptensor_offdiagonal', sample.cellid)

        row['m2_pressure_diag_0'] = m2_pressure_diag[0]
        row['m2_pressure_diag_1'] = m2_pressure_diag[1]
        row['m2_pressure_diag_2'] = m2_pressure_diag[2]
        row['m2_pressure_offdiag_0'] = m2_pressure_offdiag[0]
        row['m2_pressure_offdiag_1'] = m2_pressure_offdiag[1]
        row['m2_pressure_offdiag_2'] = m2_pressure_offdiag[2]

        #from juliacall import Main as jl
        #jl.seval("using Vlasiator")
        #meta = jl.load("filename.vlsv")
        #vcellids, vcellf = jl.readvcells(meta, int(1101), species='proton')
        #d = jl.getdensity(meta, vcellids, vcellf, species='proton')
        #v = jl.getvelocity(meta, vcellids, vcellf, species='proton')
        #p = jl.getpressure(meta, vcellids, vcellf, species='proton')

        row['pngfile'] = f'f{sample.fileid:07}c{sample.cellid:05}.png'

        if df is None:
            df = row
        else:
            df = pd.concat([df, row])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Derive values from raw samples and create CSV')
    parser.add_argument('sampledir', metavar='path', type=str, default='samples', nargs='?', help='raw samples directory')
    parser.add_argument('--datadir', type=str, default='data', nargs='?', help='vlsv data directory')
    parser.add_argument('--csvfile', type=str, default='output/features.csv', help='CSV output directory')
    args = parser.parse_args()

    vlsvdata = vlsvtools.VLSVdataset(args.datadir)

    sampler, samples = vdfsample.load_samples(args.sampledir)

    df = collect_dataframe(samples, vlsvdata)
    df.to_csv(args.csvfile, index=False)
