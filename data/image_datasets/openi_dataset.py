import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random

from data.image_datasets.data_augmentation import transforms_train_augmented

class OpenI_ImagesDataset(Dataset):
    def __init__(self, coco_dir: str, data_dir: str, visual_input_type: str, task_key: str, image_size=(384, 640), img_augmentation=False):
        '''
        Initializes an MSCOCOImagesDataset instance that handles image-side processing for VQA and other tasks that use MS-COCO images
        coco_dir: directory that contains MS-COCO data (images within 'images' folder)
        visual_input_type: format of visual input to model
        image_size: tuple indicating size of image input to model
        '''
        self.image_size = image_size
        self.img_augmentation = img_augmentation
        self.raw_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
        ])
        self.pil_transform = transforms.Resize(size=384, max_size=640)

    def get_image_data(self, image_id: str) -> Image:
        '''
        Loads image corresponding to image_id, re-sizes and returns PIL.Image object
        '''
        p =os.path.join(f'{image_id}')
        image = Image.open(p)
        image = image.convert('RGB')
        if min(list(image.size)) > 384 or hasattr(self, 'use_albef'):
            image = self.pil_transform(image)

        # Save the image before augmentation
        # output_before_path = "{}_before_augmentation.jpg".format(image_id)
        # image.save(output_before_path)
        if self.img_augmentation:
            image = transforms_train_augmented(image)
        # Save the image after augmentation
        # output_after_path = "{}_after_augmentation.jpg".format(image_id)
        # image.save(output_after_path)

        return image
