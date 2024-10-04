from torchvision.datasets import Cityscapes
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

# Define a transformation to convert PIL images to PyTorch Tensors
transform = transforms.ToTensor()

# Complete mapping of class_id to category string based on Cityscapes label documentation
CLASS_ID_TO_CATEGORY = {
    0: 'unlabeled',
    1: 'ego vehicle',
    2: 'rectification border',
    3: 'out of roi',
    4: 'static',
    5: 'dynamic',
    6: 'ground',
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle'
}

class CustomCityscapes(Cityscapes):

    def __getitem__(self, index):

        # Get data using the original __getitem__ from Cityscapes
        img, targets = super().__getitem__(index)
        
        # Convert the PIL Image to a PyTorch Tensor
        img_tensor = transform(img)

        # Process the instance ID target to extract masks
        if 'instance' in self.target_type:
            instance_index = self.target_type.index('instance')
            instance_image = targets[instance_index]
            
            # Convert to numpy array for mask processing
            masks = self.extract_instance_masks(instance_image)
        
        else:
            masks = [] # No masks if instance target is not available

        
        # Return the tensor image and the targets
        return img_tensor, masks

    def extract_instance_masks(self, instance_image):
        # Convert the instance image to a numpy array
        instance_image = np.array(instance_image)
        
        # Get unique IDs (instance IDs and class IDs encoded)
        unique_ids = np.unique(instance_image)
        
        masks = []

        # Iterate through each unique instance
        for unique_id in unique_ids:
            # Extract class_id and instance_id
            class_id = unique_id // 1000
            instance_id = unique_id % 1000

            # Skip non-instanceable classes
            if instance_id == 0:
                continue
            
            # Create binary mask for the current object
            binary_mask = (instance_image == unique_id).astype(np.uint8)  # 1 for object, 0 otherwise

            # Calculate the bounding box from the mask
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            min_y, max_y = np.where(rows)[0][[0, -1]]
            min_x, max_x = np.where(cols)[0][[0, -1]]

            # Convert bbox to tensor
            bbox = [min_x, min_y, max_x, max_y]
            bbox = torch.as_tensor(bbox, dtype=torch.int)
            
            # Map class_id to category string
            category = CLASS_ID_TO_CATEGORY.get(class_id, 'unknown')

            if category != 'car' and category != 'truck' and category != 'bus':
                continue

            # Append the mask, category, and bounding box to the list
            masks.append({
                'mask': torch.tensor(binary_mask, dtype=torch.uint8),
                'category': category,
                'bbox': bbox
            })
        
        return masks
    
# For training data
train_dataset = CustomCityscapes('../data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

# For validation data
val_dataset = CustomCityscapes('../data/cityscapes', split='val', mode='fine',
                               target_type=['instance', 'color', 'polygon'])

# For test data
test_dataset = CustomCityscapes('../data/cityscapes', split='test', mode='fine',
                                target_type=['instance', 'color', 'polygon'])
