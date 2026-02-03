import random
from torch.utils.data import DataLoader
from monai.data import Dataset, PILReader

from pytorch_lightning import LightningDataModule
from monai.transforms import Compose, ForegroundMask, KeepLargestConnectedComponent
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import glob
import json

from monai.transforms import ToTensorD, LoadImageD, ResizeD,  RandSpatialCropSamplesD
from monai.transforms import ScaleIntensityRangeD, ScaleIntensityD, SpacingD,  EnsureChannelFirstD,  CropForegroundD, RandSpatialCropD, OrientationD, EnsureTyped, RandFlipD, NormalizeIntensityD, RandScaleIntensityD, RandShiftIntensityD
from monai.transforms.transform import MapTransform

def get_transform(dict_keys, pixdim, img_size, crop=False):
    crop_ops = [
        RandSpatialCropSamplesD(keys=dict_keys, roi_size = (img_size//4, img_size//4, img_size//4), num_samples =16, random_size = False)] if crop else []

    return Compose(
            [
                # spatial transforms
                #OrientationD(keys=dict_keys, axcodes='RAI'),
                SpacingD(keys=dict_keys, pixdim=(pixdim, pixdim, pixdim)),
                ResizeD(keys=dict_keys, spatial_size=(img_size, img_size, img_size)),
                #CropForegroundD(keys=dict_keys, source_key="image"),
                ScaleIntensityD(keys=dict_keys, minv=0., maxv=1.0),
            ] + crop_ops
    )

def read_json(file):
    print(file)
    with open(file) as f:
        data = json.load(f)
    return data

class ConvertToMultiChannelBasedOnBratsClassesD(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        only_WT = True
        for key in self.keys:
            
            result = []
            if only_WT:
                # merge labels 1, 2 and 3 to construct WT
                result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            else:
                # merge label 2 and label 3 to construct TC
                result.append(torch.logical_or(d[key] == 2, d[key] == 3))
                # merge labels 1, 2 and 3 to construct WT
                result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
                # label 2 is ET
                result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

def bg_maker(img):
    transform = ForegroundMask(threshold='otsu', invert=True)
    keep_largest_cc = KeepLargestConnectedComponent(applied_labels=[1])
    gray = transform(img)
    # Keep only the largest connected component
    largest_cc = keep_largest_cc(gray)
    return largest_cc

class SiteSegDataModule2D(LightningDataModule):
    def __init__(self, batch_size: int = 2, patch_size:int=64, using_dataset_sites = [1, 2, 3, 4, 6], test_mode=False):
        super().__init__()

        self.batch_size = batch_size
        self.patch_size = patch_size
        print("Using dataset sites:", using_dataset_sites)
        
        dir_path = "/Data"

        all_files = []
        for i in using_dataset_sites:
            if test_mode:
                j = random.randint(1, 3)
                all_files += glob.glob(f"{dir_path}/dataset_site{j}.json")
            else:
                all_files += glob.glob(f"{dir_path}/dataset_site{i}.json")

        preprocess_cpu_train = Compose(
            [
                LoadImageD(keys=["image"]),
                LoadImageD(keys=["label"], reader=PILReader(converter=lambda image: image.convert("L"))),
                EnsureChannelFirstD(keys=["image", "label"]),
                CropForegroundD(keys=["image", "label"], source_key="image", select_fn=bg_maker),
                ResizeD(keys=["image"], spatial_size=(300, 300)),
                ResizeD(keys=["label"], spatial_size=(300, 300), mode="nearest-exact"),
                ScaleIntensityRangeD(keys=["image", "label"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RandSpatialCropSamplesD(keys=["image", "label"],roi_size=patch_size, num_samples=1, random_size=False),
                ToTensorD(keys=["image", "label"]),
            ]
        )

        preprocess_cpu_test = Compose(
            [
                LoadImageD(keys=["image"]),
                LoadImageD(keys=["label"], reader=PILReader(converter=lambda image: image.convert("L"))),
                EnsureChannelFirstD(keys=["image", "label"]),
                CropForegroundD(keys=["image", "label"], source_key="image", select_fn=bg_maker),
                ResizeD(keys=["image"], spatial_size=(patch_size, patch_size)),
                ResizeD(keys=["label"], spatial_size=(patch_size, patch_size), mode="nearest-exact"),
                ScaleIntensityRangeD(keys=["image", "label"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                ToTensorD(keys=["image", "label"]),
            ]
        )

        train_data = []
        val_data = []
        test_data = []
        for file in all_files:
            # load json data
            data = read_json(file)
            train_data += data['training']
            val_data += data['validation']
            test_data += data['testing']

        for file in train_data:
            file['image'] = os.path.join(dir_path, file['image'])
            file['label'] = os.path.join(dir_path, file['label'])

        for file in val_data:
            file['image'] = os.path.join(dir_path, file['image'])
            file['label'] = os.path.join(dir_path, file['label'])

        for file in test_data:
            file['image'] = os.path.join(dir_path, file['image'])
            file['label'] = os.path.join(dir_path, file['label'])

        
        self.train = Dataset(data=train_data, transform=preprocess_cpu_train)
        self.val = Dataset(data=val_data, transform=preprocess_cpu_test)
        self.test = Dataset(data=test_data, transform=preprocess_cpu_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=20)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=20)
    
class SiteSegDataModule3D(LightningDataModule):
    def __init__(self, batch_size: int = 2, patch_size:int=64, using_dataset_sites = [1, 2, 3, 4, 6]):
        super().__init__()

        self.batch_size = batch_size
        self.patch_size = patch_size
        print("Using dataset sites:", using_dataset_sites)
        
        dir_path = "Training"

        all_files = []
        for i in using_dataset_sites:
            all_files += glob.glob(f"{dir_path}/dataset_site{i}.json")

        preprocess_cpu_train = Compose(
            [
            # load 4 Nifti images and stack them together
            LoadImageD(keys=["image", "label"]),
            EnsureChannelFirstD(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesD(keys="label"),
            OrientationD(keys=["image", "label"], axcodes="RAS"),
            SpacingD(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropD(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            RandFlipD(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipD(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipD(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityD(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityD(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityD(keys="image", offsets=0.1, prob=1.0),
        ]
        )

        preprocess_cpu_test = Compose(
                [
                LoadImageD(keys=["image", "label"]),
                EnsureChannelFirstD(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesD(keys="label"),
                OrientationD(keys=["image", "label"], axcodes="RAS"),
                SpacingD(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityD(keys="image", nonzero=True, channel_wise=True),
            ]
        )

        train_data = []
        val_data = []
        test_data = []
        for file in all_files:
            # load json data
            data = read_json(file)
            train_data += data['training']
            val_data += data['validation']
            test_data += data['testing']

        for file in train_data:
            file['image'] = os.path.join(dir_path, file['image'])
            file['label'] = os.path.join(dir_path, file['label'])

        for file in val_data:
            file['image'] = os.path.join(dir_path, file['image'])
            file['label'] = os.path.join(dir_path, file['label'])

        for file in test_data:
            file['image'] = os.path.join(dir_path, file['image'])
            file['label'] = os.path.join(dir_path, file['label'])

        
        self.train = Dataset(data=train_data, transform=preprocess_cpu_train)
        self.val = Dataset(data=val_data, transform=preprocess_cpu_test)
        self.test = Dataset(data=test_data, transform=preprocess_cpu_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=20)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=20)