import monai as mn
import torch

class Transform4SegmentationBase:
    def __init__(self, IMG_SIZE):
        """
        Initializes a set of data transformations for image segmentation.

        Args:
            IMG_SIZE (int): Desired spatial size for input images.
        """
        # Define training data transformations
        self.train = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys=["img","msk"], reader="nibabelreader", ensure_channel_first=True),
            mn.transforms.SqueezeDimd(keys=["img","msk"], dim=3, update_meta=True, allow_missing_keys=False),
            mn.transforms.ResizeD(keys="img", size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE),
            mn.transforms.ResizeD(keys="msk", size_mode="longest", mode="nearest", spatial_size=IMG_SIZE),
            mn.transforms.ScaleIntensityRangeD(keys="img", a_min=-1350, a_max=150, b_min=0, b_max=1, clip=True),
            mn.transforms.ScaleIntensityRangeD(keys="msk", a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
            mn.transforms.SpatialPadD(keys=["img","msk"], spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
            mn.transforms.RandFlipD(keys=["img","msk"], spatial_axis=0, prob=0.5),
            mn.transforms.RandFlipD(keys=["img","msk"], spatial_axis=1, prob=0.5),
            mn.transforms.RandGaussianNoiseD(keys=["img","msk"], mean=0.0, std=0.3, prob=0.5),
            mn.transforms.RandAffineD(keys=["img","msk"], mode="bilinear", prob=0.5, rotate_range=0.4, scale_range=0.1, translate_range=IMG_SIZE//20, padding_mode="border"),
            mn.transforms.SelectItemsD(keys=["img","msk"]),
            mn.transforms.ToTensorD(keys=["img","msk"], dtype=torch.float, track_meta=False)])

        # Define validation data transformations
        self.val = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys=["img","msk"], reader="nibabelreader", ensure_channel_first=True),
            mn.transforms.SqueezeDimd(keys=["img","msk"], dim=3, update_meta=True, allow_missing_keys=False),
            mn.transforms.ResizeD(keys="img", size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE),
            mn.transforms.ResizeD(keys="msk", size_mode="longest", mode="nearest", spatial_size=IMG_SIZE),
            mn.transforms.ScaleIntensityRangeD(keys="img", a_min=-1350, a_max=150, b_min=0, b_max=1, clip=True),
            mn.transforms.ScaleIntensityRangeD(keys="msk", a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
            mn.transforms.SpatialPadD(keys=["img","msk"], spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
            mn.transforms.SelectItemsD(keys=["img","msk"]),
            mn.transforms.ToTensorD(keys=["img","msk"], dtype=torch.float, track_meta=False)])
