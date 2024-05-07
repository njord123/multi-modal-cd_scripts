# lager en dataloader og kjører den på trening og test datasettene

import os
import glob
import torch
import torch.utils.data as data
import torchvision.transforms as T
import rasterio
import utils


class TrainLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path, mean=None, std=None):
        
        if mean == None or std==None:
            _, mean, std = utils.img_stats(os.path.join(folder_path, "img"))
        self.transform = T.Normalize(mean=mean,std=std)

        super(TrainLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, 'img', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'mask', os.path.basename(img_path)))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = rasterio.open(img_path).read()
            label = rasterio.open(mask_path).read()
            return {'image': utils.normalize(self.transform(torch.nan_to_num(
                      torch.from_numpy(data.astype("float")).to(torch.float)
                      ))),
                    'mask': torch.from_numpy(label.astype("uint8"))[0,...].to(torch.long)}

    def __len__(self):
        return len(self.img_files)


def make_trainloaders(path, mean=None, std=None):

    dataset = TrainLoaderSegmentation(folder_path=path, mean=mean, std=std)
    print('dataset size:', len(dataset))
 
    return dataset