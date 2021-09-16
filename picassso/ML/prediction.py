from speedrun import BaseExperiment, TensorboardMixin
from argparse import Namespace
import torch

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from scipy.misc import imsave
from . import SDSSUNet

# make this comply with the package definitions of where the functions aare defined
from datasets import SyntheticSDSS, load_halo_properties
from shutil import copyfile
import numpy as np

from sunpy.sunpy__synthetic_image import congrid
from tqdm import tqdm

from torch import load
import h5py

from training import CropedLoss
from util import from_logspace

import pathlib


def save_image(tensor, filename):
    image = make_grid(tensor[0, :3].permute(1, 0, 2, 3), normalize=True)
    image = image.cpu().numpy().transpose(1, 2, 0)
    imsave(filename, image)

# means = {"gas_GFM_Metallicity": (-5.523196370732448, 1.3777920023182069),
#               "gas_Masses": (-10.215950934530275, 1.4487507455034856),
#               "stars_GFM_Metallicity": (-5.678650891964106, 1.8846359377870625),
#               "stars_GFM_StellarFormationTime": (-1.754890754639367, 1.2792070463405334),
#               "stars_Masses": (-10.250709358845338, 1.6992289685274156)}

if __name__ == '__main__':

    ds = "val"
    root_folder = 'experiments/ubn_06/'
    LOGSPACE = False


    # for t in [5000, 20000, 50000, 130000]:
    for t in [100000]:


        # _, halodata_keys = ['gas_GFM_Metallicity', 'gas_Masses', 'gas_density', 'star_density', 'stars_GFM_Metallicity', 'stars_GFM_StellarFormationTime', 'stars_Masses']

        halodata_keys = ['gas_GFM_Metallicity', 'gas_Masses', 'gas_NeutralHydrogenAbundance', 'gas_StarFormationRate',
                         'stars_GFM_Metallicity', 'stars_GFM_StellarFormationTime', 'stars_Masses']

        # model = load(f"{root_folder}/Weights/checkpoint_iter_{t}.pt")['model']
        model = load(f"{root_folder}/best_checkpoint.pytorch")['_model']
        loader = SyntheticSDSS(f"./data/directory_catalog_{ds}_s.txt",
                               normalization='div',
                               smoothing=0.)
        loader.random_camera = False
        loader.image_synth_thresh = 0.0

        output_folder = f"{root_folder}/prediction/{ds}_{t}/"
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for idx in tqdm(range(len(loader)), dynamic_ncols=True, ascii=True):
                halo_id = loader.catalog[idx]

                inp, gt = loader[idx]
                # print(idx, )
                loader.current_psize

                tinp = torch.from_numpy(inp[None]).cuda()
                gt = torch.from_numpy(gt[None]).cpu().numpy()
                prediction = model(tinp).cpu().numpy()

                # weight = torch.ne(target.sum(dim=1), 0.)[:, None].float()
                # smooth_weights = scipy.ndimage.filters.gaussian_filter
                with h5py.File(f"{output_folder}/halo_{halo_id}_camera_{0}_input_data.h5", "w") as h5file:
                    h5file.create_dataset("gt", data=gt)
                    h5file.create_dataset("inp", data=inp)
                    h5file.create_dataset("pred", data=prediction[0, :].copy())

                pathlib.Path(f'{output_folder}/true/').mkdir(parents=True, exist_ok=True)
                # TODO: just copy the file instead of the data!
                with h5py.File(f"{output_folder}/true/halo_{halo_id}_camera_{0}_physical_data.h5", "w") as h5file:
                    for c, key in enumerate(halodata_keys):
                        if LOGSPACE:
                            h5file.create_dataset(key, data=congrid(from_logspace(gt[0, c]), (256, 256)))
                        else:
                            h5file.create_dataset(key, data=gt[0, c])
                        h5file[key].attrs["psize"] = loader.current_psize



                with h5py.File(f"{output_folder}/halo_{halo_id}_camera_{0}_physical_data.h5", "w") as h5file:
                    # weight = (torch.gt(gt.max(dim=1)[0], 0.)[:, None].float()).numpy()
                    # for c in range(prediction.shape[1]):
                    for c, key in enumerate(halodata_keys):

                        predn = prediction[0, c].copy()

                        if LOGSPACE:
                            # transform from logspace to regular space and rebin to 256,256
                            predn = congrid(from_logspace(predn), (256, 256))


                        # gt_n = gt[0, c].copy()
                        # plt.hist(predn.ravel(), alpha=0.5, label='prediction')
                        # plt.hist(gt_n.ravel(), alpha=0.5, label='gt')
                        # plt.savefig(f"{output_folder}/norm_prediction_{halo_id}_{c}.png")
                        # plt.close()

                        # plt.hist((prediction[:, c] - gt[:, c].cpu().numpy())[weight[:, 0]>0].ravel())
                        # plt.savefig(f"diff_{c}.png")
                        # plt.close()

                        h5file.create_dataset(key, data=predn)
                        h5file[key].attrs["psize"] = loader.current_psize
