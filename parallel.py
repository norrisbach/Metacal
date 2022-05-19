#!/usr/bin/env python
import sys
import ngmix
import galsim
import numpy as np
from schwimmbad import MPIPool

def biasFunc(x, m, c):
    '''Systematic bias of shear estimation
    Parameters:
    ----
    x:  input shear
    m:  multiplicative bias
    c:  additive bias

    Returns:
    ----
    y:  estimated shear
    '''
    y   =   (1+m) * x + c
    return y

def select(data, shear_type):
    """
    select the data by shear type and size
    Parameters
    ----------
    data: array
        The array with fields shear_type and T
    shear_type: str
        e.g. 'noshear', '1p', etc.
    Returns
    -------
    array of indices
    """

    w, = np.where(
        (data['flags'] == 0) & (data['shear_type'] == shear_type)
    )
    return w

def make_struct(res, obs, shear_type):
    """
    make the data structure
    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', and 'T'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type
    Returns
    -------
    1-element array with fields
    """
    dt = [
        ('flags', 'i4'),
        ('shear_type', 'U7'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('T', 'f8'),
        ('Tpsf', 'f8'),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['flags'] = res['flags']
    if res['flags'] == 0:
        #data['s2n'] = res['s2n']
        data['s2n'] = res['s2n']
        # for moments we are actually measureing e, the elliptity
        data['g'] = res['e']
        data['T'] = res['T']
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['T'] = np.nan
        data['Tpsf'] = np.nan

        # we only have one epoch and band, so we can get the psf T from the
        # observation rather than averaging over epochs/bands
        data['Tpsf'] = obs.psf.meta['result']['T']
    return data

def make_data(rng, FbyB, shear, version=0, SbyN=20):
    """
    simulate an exponential object with moffat psf
    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    noise: float
        Noise for the image
    FbyB: float
        source by background ratio
    shear: (g1, g2)
        The shear in each component
    Returns
    -------
    ngmix.Observation
    """

    scale    = 0.263
    psf_fwhm = 0.9
    gal_hlr  = 0.5
    dy, dx   = rng.uniform(low=-scale/2, high=scale/2, size=2)

    psf = galsim.Moffat(beta=2.5, fwhm=psf_fwhm,
    ).shear(g1=0.02, g2=-0.01,)


    obj0 = galsim.Exponential(
        half_light_radius=gal_hlr, flux=125e3
    ).shear(
        g1=shear,
        g2=0,
    ).shift(dx=dx, dy=dy,)
    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale = scale)

    # psf noise
    psf_noise= 1e-9
    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    radius = 10

    ngrid = im.array.shape[0]
    flux_tmp = np.sum((im.array)[ngrid//2-radius:ngrid//2+radius+1, ngrid//2-radius:ngrid//2+radius+1])
    F = SbyN**2.*(1+FbyB)/FbyB
    B = F/FbyB
    B_val = B/(2.*radius+1)**2.
    F_val= F/(2.*radius+1)**2.

    im = (im/flux_tmp)*F

    if version==0:
        noise_image = rng.normal(scale=1, size=im.array.shape)
        noise_image *= np.sqrt(B_val + im.array)
    elif version==1:
        noise_image = rng.normal(scale=1, size=im.array.shape)
        noise_image *= np.sqrt(B_val+F_val)
    else:
        raise ValueError('version should be 0 or 1')
    variance_array = np.ones_like(im.array)*(B_val+F_val)
    im += noise_image
    wt = 1.0/variance_array
    imArr = im.array

    cen = (np.array(imArr.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0
    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy/scale, col=cen[1] + dx/scale, scale=scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
    )

    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )
    obs = ngmix.Observation(
        imArr,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )
    return obs

def analyze(info):
    """
    element = (num_gals, FBratioArr[n], g_true, seed, version, first, SbyN)
    """
    if ~info[4]:
        print('Analyzing for F/B= %.2f' %info[1])
    else:
        print('Analyzing for F/B= 0')
    data = []
    x = []
    y = []
    s2nArr=[]
    shear_error = []

    for i in range(len(info[2])):
        seedNum = info[3][i]
        rng = np.random.RandomState(seedNum)
        shearIn = info[2][i]
        print('Analyzing for shear= %.2f' %shearIn)

        dlist = []

        for _ in range(info[0]):
            imgdata = make_data(rng=rng, FbyB=info[1], shear=shearIn, version=info[4], SbyN=info[6])
            obs = imgdata

            resdict, obsdict = boot.go(obs)
            for stype, sres in resdict.items():
                st = make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
                dlist.append(st)
                del st
            del obs, resdict, obsdict

        data = np.hstack(dlist)
        del dlist

        w = select(data=data, shear_type='noshear')
        w_1p = select(data=data, shear_type='1p')
        w_1m = select(data=data, shear_type='1m')
        g_1p = data['g'][w_1p, 0].mean()
        g_1m = data['g'][w_1m, 0].mean()
        R11 = (g_1p - g_1m)/0.02
        s2n = data['s2n'].mean()

        g = data['g'][w].mean(axis=0)
        shear = g / R11

        g_error = data['g'][w].std(axis=0) / np.sqrt(w.size)
        shear_error.append(g_error[0]/R11)

        x.append(info[1])
        y.append(shear[0])
        s2nArr.append(s2n)
        del data
    return (x, y, shear_error, s2nArr)


rng = np.random.RandomState(1001)

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
    rng=rng,
    psf='gauss',
    types=['noshear', '1p', '1m'],
)

#No work
FBratioArr = np.logspace(start=-2, stop=2, num=5, base=10.0)
num_gals = 100
g_true= [-0.03, -0.01, 0, 0.01, 0.03]
seed = [1, 2, 3, 4, 5]
version = 0
first = False
SbyN = 20
data=[]

# preparation of info
for n in range(5):
    element = (num_gals, FBratioArr[n], g_true, seed, version, first, SbyN)
    data.append(element)

pool = MPIPool()
print('I am running')
if not pool.is_master():
    pool.wait()
    sys.exit(0)
pool.map(analyze,data)
pool.close()
