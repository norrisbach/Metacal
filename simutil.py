import galsim
import numpy as np

def make_data(rng,shear,scale):
    """
    simulate an exponential object with moffat psf
    Parameters
    ----
    rng: np.random.RandomState
        The random number generator
    shear: (g1, g2)
        The shear in each component
    Results
    ----
    im:  np.ndarray
        galaxy image
    psf_im: np.ndarray
        psf image
    """

    psf_fwhm = 0.9
    gal_hlr  = 0.5
    dy, dx   = rng.uniform(low=-scale/2, high=scale/2, size=2)

    psf = galsim.Moffat(beta=2.5, fwhm=psf_fwhm,
    ).shear(g1=0.02, g2=-0.01,)


    obj0 = galsim.Exponential(
        half_light_radius=gal_hlr, flux=125e3
    ).shear(
        g1=shear[0],
        g2=shear[1],
    ).shift(dx=dx, dy=dy,)
    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale = scale).array
    return im,psf_im

def add_noise(im0,psf_im0,scale,rng,noise,version=0,first=False):

    """
    add noise to galaxy and psf image
    Parameters
    ----
    im0:  np.ndarray
        noiseless galaxy image
    psf_im0: np.ndarray
        noiseless psf image
    rng: np.random.RandomState
        The random number generator
    noise: float
        Noise for the image
    Results
    ----
    galdict: dict
        a dictionary with galaxy info
    psfdict: dict
        a dictionary with psf info
    background: list
        a list of noise variance
    """

    # PSF
    # psf noise
    psf_noise= 1e-10
    psf_im =  psf_im0.copy()
    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    psf_wt = 1.0/psf_noise**2


    # output dictionary for psf
    psfdict={
        'im': psf_im,
        'wt': psf_wt,
    }

    # Gal
    # gal noise
    im     =  im0.copy()
    random = galsim.BaseDeviate()
    if version not in [0,1]:
        # 0 for background noise only
        # 1 for background and source noise
        raise ValueError('version can only be integer 0 or 1')
    backgrounds = []
    var_image =   im*version + (noise)**2
    wt    =  1./var_image # weight is just the inverse of var map
    if first:
        backgrounds.append(var_image)
    varNoise  =   galsim.VariableGaussianNoise(random, var_image)
    nx=var_image.shape[1];ny=var_image.shape[0]
    noise_gim =  galsim.ImageF(nx,ny,scale=scale)
    # add noise
    noise_gim.addNoise(varNoise)
    im += noise_gim.array

    # output dictionary for psf
    galdict={
        'im': im,
        'wt': wt,
    }
    return galdict, psfdict, backgrounds

try:
    import ngmix
    def make_ngmix_data(galdict,psfdict,scale):
        '''
        Parameters
        ----
        galdict: dict
            a dictionary with galaxy info
        psfdict: dict
            a dictionary with psf info
        Returns
        ----
        ngmix.Observation
        '''

        # psf Jacobian
        psf_cen = (np.array(psfdict['im'].shape)-1.0)/2.0
        psf_jacobian = ngmix.DiagonalJacobian(
            row=psf_cen[0], col=psf_cen[1], scale=scale,
        )
        psf_obs = ngmix.Observation(
            psfdict['im'],
            weight=psfdict['wt'],
            jacobian=psf_jacobian,
        )

        # gal Jacobian
        cen = (np.array(galdict['im'].shape)-1.0)/2.0
        jacobian = ngmix.DiagonalJacobian(
            row=cen[0], col=cen[1], scale=scale,
        )
        obs = ngmix.Observation(
            galdict['imArr'],
            weight=galdict['wt'],
            jacobian=jacobian,
            psf=psf_obs,
        )
        return obs
except (ImportError, KeyError) as error:
    with_hst=False
