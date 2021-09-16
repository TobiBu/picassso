from speedrun import BaseExperiment, TensorboardMixin, HyperoptMixin
from argparse import Namespace
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from scipy.misc import imsave
from model import SDSSUNet
from datasets import SyntheticSDSS
from shutil import copyfile
from torch import load
import sys
from time import sleep


class CropedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = MSELoss()

    def forward(self, preds, targets):
        return self.loss(preds[:, :, 4:-4, 4:-4], targets)


class Maiden(BaseExperiment, HyperoptMixin, TensorboardMixin):

    def __init__(self, experiment_directory):
        super(Maiden, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = [
            'data_loader', '_device']
        self.read_config_file()

    def build_model(self):
        return SDSSUNet(**self.get('model/kwargs'))

    def inferno_build_criterion(self):
        print("Using criterion ", self.get('trainer/criterion'))
        if self.get('trainer/criterion') == "CropedLoss":
            self._trainer.build_criterion(CropedLoss())
        else:
            # use inferno default losses
            super().inferno_build_criterion()

    def build_train_loader(self):
        return DataLoader(SyntheticSDSS(**self.get('loader/data_kwargs')),
                                        **self.get('loader/loader_kwargs'))

    def build_val_loader(self):
        return DataLoader(SyntheticSDSS(**self.get('val_loader/data_kwargs')),
                                        **self.get('val_loader/loader_kwargs'))

if __name__ == '__main__':

    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    else:
        folder_name = "dev_00"

    project_directory = f'experiments/{folder_name}/'
    if not os.path.exists(project_directory):
        print("creating new project directory ", project_directory)
        os.mkdir(project_directory)
        print("copying template config")
        os.mkdir(project_directory + "/Configurations/")
        print(project_directory +
              "/Configurations/train_config.yml")
        copyfile("configs/train_config.yml", project_directory +
                 "/Configurations/train_config.yml")

    exp = Maiden(project_directory)
    exp.train()
