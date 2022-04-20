# see https://schwimmbad.readthedocs.io/en/latest/examples/
# for reference
# The following command uses 8 cores to process reference Id from 0 to 100
# mpirun -np 8 ./example_parallel.py --minId 0 --maxId 100
import os
import gc
import sys
import ngmix
import numpy as np
import argparse
from schwimmbad import MPIPool

def metacal_init(rg):
    # We will measure moments with a fixed gaussian weight function
    weight_fwhm= 1.2
    fitter     = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

    # these "runners" run the measurement code on observations
    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
    runner     = ngmix.runners.Runner(fitter=fitter)

    # this "bootstrapper" runs the metacal image shearing as well as both psf
    # and object measurements
    #
    # We will just do R11 for simplicity and to speed up this example;
    # typically the off diagonal terms are negligible, and R11 and R22 are
    # usually consistent
    boot      = ngmix.metacal.MetacalBootstrapper(
        runner= runner, psf_runner=psf_runner,
        rng= rg,
        psf='gauss',
        types=['noshear', '1p', '1m'],
    )
    return boot


def do_process(iref):
    rng = np.random.RandomState(iref)
    boot=metacal_init(rng)


    del rng
    return

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--minId', required=True,type=int,
                        help='minimum id number, e.g. 0')
    parser.add_argument('--maxId', required=True,type=int,
                        help='maximum id number, e.g. 13*108=1404')
    args = parser.parse_args()
    # create a list of reference ID
    refs    =   list(range(args.minId,args.maxId))

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    pool.map(do_process,refs)
    pool.close()
    return

if __name__=='__main__':
    main()

