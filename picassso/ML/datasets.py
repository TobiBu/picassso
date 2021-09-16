


#########################
# This whole file might be obsolete except for the synthetic image routine.
# check and then move some stuff to the galaxy classes...
##########################




from torch.utils.data import Dataset
from inferno.io.transform import Compose
# from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.image import RandomFlip, RandomCrop
from inferno.io.transform.image import RandomRotate, ElasticTransform
import numpy as np
from sunpy.sunpy__synthetic_image import build_synthetic_image_with_scale, congrid
from api_keys import illustris_key
import os
import random
import h5py
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import block_reduce

from util import open_hdf5, load_halo_properties


def load_catalog(halo_idx_file, halo_path='./data/halodata', fits_path='./data'):
        # load halo index file
    catalog = np.loadtxt(halo_idx_file,
                         dtype={'names': ('subdirs', 'galaxy_numbers', 'galaxy_masses'),
                                'formats': ('S3', 'i8', 'f8')})['galaxy_numbers']
    # filter catalog if physical properties are not available
    # remove this once Tobi has computed everything
    mask = []
    for halo_idx in catalog:
        target_file_name = f'{halo_path}/halo_{halo_idx}_camera_0_physical_data.h5'
        fits_filename = f"{fits_path}/broadband_{halo_idx}.fits"
        if not os.path.isfile(fits_filename):
            get_fits_file(halo_idx)
        mask.append((os.path.isfile(target_file_name) and
                     os.path.isfile(fits_filename)))
        if not (os.path.isfile(target_file_name) and
                os.path.isfile(fits_filename)):
            print(halo_idx, os.path.isfile(target_file_name), os.path.isfile(fits_filename))
    mask = np.array(mask, dtype=bool)
    catalog = catalog[mask]
    print("catalog size = ", mask.sum(), " / ", mask.size)
    return catalog


class SyntheticSDSS(Dataset):
    """Generate synthetic SDSS Images using sunpy"""

    def __init__(self, halo_idx_file='./data/directory_catalog_135_train.txt',
                 normalization='log', smoothing=2., padding=False,
                 predict_logspace=True, downsample_factor=0):

        self.normalization = normalization
        self.smoothing = smoothing
        self.padding = padding
        self.random_camera = True
        self.image_synth_thresh = 0.05
        self.predict_logspace = predict_logspace
        self.downsample_factor = downsample_factor

        # do not normalize for now
        self.means = {}
        # {"gas_GFM_Metallicity": (-5.523196370732448, 1.3777920023182069),
        #   "gas_Masses": (-10.215950934530275, 1.4487507455034856),
        #   "stars_GFM_Metallicity": (-5.678650891964106, 1.8846359377870625),
        #   "stars_GFM_StellarFormationTime": (-1.754890754639367, 1.2792070463405334),
        #   "stars_Masses": (-10.250709358845338, 1.6992289685274156)}

        self.catalog = load_catalog(halo_idx_file)

        # data augmentation
        self.augment = Compose(RandomFlip(),
                               RandomRotate())

        self.synthesis_args = {
            # default option = False, turn on one-by-one
            'add_background':       True,
            'add_noise':            False,  # True,
            'add_psf':              True,
            'rebin_phys':           True,
            'resize_rp':            False,          # deactivate this for now
                                                    # activation would lead to an internal crop,
                                                    # that needs to be applied to the gt halo properties
            'rebin_gz':             True,           # always true, all pngs 424x424
            'scale_min':            1e-10,          # was 1e-4
            'lupton_alpha':         2e-12,          # was 1e-4
            'lupton_Q':             10,             # was ~100
            'pixelsize_arcsec':     0.24,
            'psf_fwhm_arcsec':      1.0,
            'sn_limit':             25.0,
            'sky_sig':              None,           #
            'redshift':             0.05,           #
            'b_fac':                1.1,
            'g_fac':                1.0,
            'r_fac':                0.9,
            'camera':               0,
            'seed_boost':           1.0,
            'save_fits':            False
        }

        self.accepted_keys = ['gas_GFM_Metallicity',
                              'gas_Masses',
                              'gas_NeutralHydrogenAbundance',
                              'gas_StarFormationRate',
                              'stars_GFM_Metallicity',
                              'stars_GFM_StellarFormationTime',
                              'stars_Masses']
        os.makedirs("./data/", exist_ok=True)

    def __getitem__(self, index):
        # create synthetic SDSS image
        halo_idx = self.catalog[index]
        camera = 0
        # draw one of 4 random camera positions
        if self.random_camera:
            camera = random.randint(0, 3)

        self.synthesis_args["camera"] = camera

        # generate new image if old file is not available
        # or randomly with image_synth_thresh chance
        synth_new_image = np.random.rand() < self.image_synth_thresh
        if (can_load_synth_image(halo_idx, camera) and not synth_new_image):
            try:
                img = load_synth_image(halo_idx, camera)
            except NameError:
                print("replacing potentially corrupted file",
                      synth_file_name(halo_idx, camera))
                fits_file = get_fits_file(halo_idx)
                img = synthesise_ugriz_img(fits_file, **self.synthesis_args)
                save_synth_image(img, halo_idx, camera)
        else:
            fits_file = get_fits_file(halo_idx)
            img = synthesise_ugriz_img(fits_file, **self.synthesis_args)
            # overwrite/create target file
            save_synth_image(img, halo_idx, camera)

        img = self.normalize_image(img)

        # load target image
        gt, keys, self.current_psize = load_halo_properties(halo_idx,
                                                            camera,
                                                            img.shape[1:],
                                                            smoothing=self.smoothing,
                                                            log=self.predict_logspace)

        mask = []

        for i, key in enumerate(keys):
            if key in self.means:
                gt[i] -= self.means[key][0]
                gt[i] /= self.means[key][1]
            mask.append(key in self.accepted_keys)

        gt = gt[mask]

        if self.padding:
            img = np.pad(img, ((0, 0), (4, 4), (4, 4)), 'reflect')

        img, gt = img.astype(np.float32), gt.astype(np.float32)
        img, gt = self.augment(img, gt)

        if self.downsample_factor > 0:
            df = self.downsample_factor
            img = block_reduce(img, (1, df, df), func=np.mean)
            gt = block_reduce(gt, (1, df, df), func=np.mean)

        return img, gt

    def normalize_image(self, image_data):

        if self.normalization == 'sinh':
            # plot normalization
            lupton_alpha = 0.5
            lupton_Q = 0.5
            scale_min = 1e-4,
            I = image_data.sum(axis=0)
            val = np.arcsinh(lupton_alpha * lupton_Q * (I - scale_min)) / lupton_Q
            I[I < 1e-6] = 1e100        # from below, this effectively sets the pixel to 0
            image_data *= (val / I)
            image_data /= 50.
        elif self.normalization == 'log':
            # make all values strictly positive
            np.clip(image_data, 1e-4, None, out=image_data)
            image_data = np.log(image_data)
        elif self.normalization == 'div':
            image_data /= 1e13
        else:
            pass

        return image_data

    def __len__(self):
        return len(self.catalog)


def get_fits_file(halo_idx):
    cmd = 'wget --content-disposition --directory-prefix=./data/  --header="API-Key: ' + \
        illustris_key + '" "' + "http://www.illustris-project.org" + \
        '/api/Illustris-1/snapshots/135/subhalos/' + str(halo_idx) +  \
        '/stellar_mocks/broadband.fits"'

    filename = "./data/broadband_" + str(halo_idx) + ".fits"

    if(not (os.path.isfile("./data/broadband_" + str(halo_idx) + ".fits"))):
        os.system(cmd)
    return filename


def synth_file_name(halo_idx, camera):
    return f'./data/halodata/halo_{halo_idx}_camera_{camera}_synthetic_image.h5'


def can_load_synth_image(halo_idx, camera):
    image_file_name = synth_file_name(halo_idx, camera)
    return os.path.isfile(image_file_name)


def load_synth_image(halo_idx, camera):
    image_file_name = synth_file_name(halo_idx, camera)
    with open_hdf5(image_file_name, "r") as h5file:
        img = h5file['data'].value
    return img


def save_synth_image(img, halo_idx, camera):
    image_file_name = synth_file_name(halo_idx, camera)
    with open_hdf5(image_file_name, "w") as h5file:
        img = h5file.create_dataset('data', data=img)


def synthesise_ugriz_img(filename, camera, **kwargs):

    seed = random.randrange(1e7)
    image_data = []

    # calcualte the Petro Radius once
    r_petro_kpc = None

    # resize to Galaxy Zoo resolution
    kwargs['camera'] = camera

    for channel in "ugriz":
        img, r_petro_kpc, _ = build_synthetic_image_with_scale(filename,
                                                               channel + '_SDSS.res',
                                                               seed=seed,
                                                               r_petro_kpc=r_petro_kpc,
                                                               fix_seed=False,
                                                               **kwargs)
        image_data.append(img)

    image_data = np.stack(image_data)

    return image_data


if __name__ == '__main__':
    ds = SyntheticSDSS()
    img, gt = ds[3]
    print(img.min(), img.max())
