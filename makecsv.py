import os
import logging
import numpy as np
import scipy.stats as scs
import pandas as pd

import vlsvtools
import vdfsample

from juliacall import Main as jl

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
            row['rho'] = cell.vlsvfile.handle.read_variable('rho', sample.cellid)
        if f'{cell.populations[0]}/vg_rho' in cell.vlsvfile.handle.get_all_variables():
            row['rho'] = cell.vlsvfile.handle.read_variable(f'{cell.populations[0]}/vg_rho', sample.cellid)

        jl.seval("using Vlasiator")
        meta = jl.load(cell.vlsvfile.filename)
        vcellids, vcellf = jl.readvcells(meta, int(sample.cellid), species=cell.populations[0])

        m0_density = jl.getdensity(meta, vcellids, vcellf, species=cell.populations[0])

        m1_velocity = jl.getvelocity(meta, vcellids, vcellf, species=cell.populations[0])
        row['m1_velocity_0'] = m1_velocity[0]
        row['m1_velocity_1'] = m1_velocity[1]
        row['m1_velocity_2'] = m1_velocity[2]

        m2_pressure = jl.getpressure(meta, vcellids, vcellf, species=cell.populations[0])
        row['m2_pressure_0'] = m2_pressure[0]
        row['m2_pressure_1'] = m2_pressure[1]
        row['m2_pressure_2'] = m2_pressure[2]
        row['m2_pressure_3'] = m2_pressure[3]
        row['m2_pressure_4'] = m2_pressure[4]
        row['m2_pressure_5'] = m2_pressure[5]

        row['m0_density'] = m0_density
        row['m1_velocity'] = np.linalg.norm(m1_velocity)
        row['m2_pressure'] = np.linalg.norm(m2_pressure)

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
