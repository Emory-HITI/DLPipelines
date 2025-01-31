import monai as mn
import torch

# class Transform4ClassifierBase:
#     def __init__(self, IMG_SIZE):
#         """
#         Initializes a set of data transformations for image classification.

#         Args:
#             IMG_SIZE (int): Desired spatial size for input images.
#         """
#         # Define the training data transformations
#         self.train = mn.transforms.Compose([
#             mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
#             #mn.transforms.SqueezeDimd(keys="img", dim=3, update_meta=True, allow_missing_keys=False), # comment this if using png
#             #mn.transforms.HistogramNormalized(keys="img"),
#             mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
#             mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
#             mn.transforms.SpatialPadD(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
#             mn.transforms.RandFlipD(keys="img", spatial_axis=0, prob=0.2),
#             mn.transforms.RandFlipD(keys="img", spatial_axis=1, prob=0.2),
#             mn.transforms.RandGaussianNoiseD(keys="img", mean=0.0, std=0.3, prob=0.5),
#             mn.transforms.RandAffineD(keys="img", mode="bilinear", prob=0.5, rotate_range=0.4, scale_range=0.1, translate_range=IMG_SIZE//20, padding_mode="border"),
#             mn.transforms.SelectItemsD(keys=["img", "label", "paths"]),
#             mn.transforms.ToTensorD(keys="img", dtype=torch.float, track_meta=False),
#             mn.transforms.ToTensorD(keys=["label"], dtype=torch.float)])

#         # Define the validation data transformations
#         self.val = mn.transforms.Compose([
#             mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
#             #mn.transforms.SqueezeDimd(keys="img", dim=3, update_meta=True, allow_missing_keys=False),
#             #mn.transforms.HistogramNormalized(keys="img"),
#             mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
#             mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
#             mn.transforms.SpatialPadD(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
#             mn.transforms.SelectItemsD(keys=["img", "label", "paths"]),
#             mn.transforms.ToTensorD(keys="img", dtype=torch.float, track_meta=False),
#             mn.transforms.ToTensorD(keys=["label"], dtype=torch.float)])
        
#         self.predict = mn.transforms.Compose([
#             mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
#             #mn.transforms.SqueezeDimd(keys="img", dim=3, update_meta=True, allow_missing_keys=False),
#             #mn.transforms.HistogramNormalized(keys="img"),
#             mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
#             mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
#             mn.transforms.SpatialPadD(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
#             mn.transforms.SelectItemsD(keys=["img", "paths"]),
#             mn.transforms.ToTensorD(keys="img", dtype=torch.float, track_meta=False)])



class Transform4ClassifierBase:
    def __init__(self, IMG_SIZE, CLASSES):
        """
        Initializes a set of data transformations for image classification.

        Args:
            IMG_SIZE (int): Desired spatial size for input images.
        """
        self.CLASSES = CLASSES
        
        # Define the training data transformations
        self.train = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
            #mn.transforms.SqueezeDimd(keys="img", dim=3, update_meta=True, allow_missing_keys=False),
            #mn.transforms.HistogramNormalized(keys="img"),
            mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
            mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
            mn.transforms.SpatialPadD(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
            mn.transforms.RandFlipD(keys="img", spatial_axis=0, prob=0.2),
            mn.transforms.RandFlipD(keys="img", spatial_axis=1, prob=0.2),
            mn.transforms.RandGaussianNoiseD(keys="img", mean=0.0, std=0.3, prob=0.5),
            mn.transforms.RandAffineD(keys="img", mode="bilinear", prob=0.5, rotate_range=0.4, scale_range=0.1, translate_range=IMG_SIZE//20, padding_mode="border"),

            mn.transforms.ToTensorD(keys=[*CLASSES], dtype=torch.float),            
            mn.transforms.ConcatItemsD(keys=[*CLASSES], name='cls'),
            
            mn.transforms.SelectItemsD(keys=["img", "cls", "paths"]),
            mn.transforms.ToTensorD(keys="img", dtype=torch.float, track_meta=False),
            mn.transforms.ToTensorD(keys=["cls"], dtype=torch.float)])

        # Define the validation data transformations
        self.val = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
            #mn.transforms.SqueezeDimd(keys="img", dim=3, update_meta=True, allow_missing_keys=False),
            #mn.transforms.HistogramNormalized(keys="img"),
            mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
            mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
            mn.transforms.SpatialPadD(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),

            mn.transforms.ToTensorD(keys=[*CLASSES], dtype=torch.float),            
            mn.transforms.ConcatItemsD(keys=[*CLASSES], name='cls'),
            
            mn.transforms.SelectItemsD(keys=["img", "cls", "paths"]),
            mn.transforms.ToTensorD(keys="img", dtype=torch.float, track_meta=False),
            mn.transforms.ToTensorD(keys=["cls"], dtype=torch.float)])

        self.predict = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
            #mn.transforms.SqueezeDimd(keys="img", dim=3, update_meta=True, allow_missing_keys=False),
            #mn.transforms.HistogramNormalized(keys="img"),
            mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
            mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
            mn.transforms.SpatialPadD(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
            mn.transforms.SelectItemsD(keys=["img", "paths"]),
            mn.transforms.ToTensorD(keys="img", dtype=torch.float, track_meta=False)])
