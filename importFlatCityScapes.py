from torchvision.datasets import Cityscapes
from torchvision import transforms
import torch
# Define a transformation to convert PIL images to PyTorch Tensors
transform = transforms.ToTensor()

class CustomCityscapes(Cityscapes):

    def __init__(self, root, split='train', mode='fine', target_type=['instance', 'color', 'polygon']):
        
        super().__init__(root, split=split, mode=mode, target_type=target_type)
        self.data = []
        self.load_data()

    def load_data(self):

        for index in range(len(self)):
            img, targets = super().__getitem__(index)
            img_tensor = transform(img)

            if 'polygon' in self.target_type:
                poly_index = self.target_type.index('polygon')
                polygons = targets[poly_index]

                for polygon in polygons['objects']:
                    bbox, polygon_points, label = self.polygon_to_bbox(polygon)
                    self.data.append({"image": img_tensor,"gt_bbox": bbox,"gt_polygon": polygon_points,"gt_str_label": label})

    def __getitem__(self, idx):
        # Return a single data item containing an image patch and its bbox
        return self.data[idx]

    def __len__(self):
        if self.split == 'train':
            return 2975
        if self.split == 'val':
            return 500
        if self.split == 'test':
            return 1525
        return 0

    def polygon_to_bbox(self, polygon):

        # Calculate bounding boxes from polygons
        polygon_points = polygon['polygon']
        min_x = min(point[0] for point in polygon_points)
        max_x = max(point[0] for point in polygon_points)
        min_y = min(point[1] for point in polygon_points)
        max_y = max(point[1] for point in polygon_points)
        bbox = [min_x, min_y, max_x, max_y]
        label = polygon['label']
        bbox = torch.as_tensor(bbox, dtype=torch.int)
        return bbox, polygon_points, label

# For training data
train_dataset = CustomCityscapes('./data/cityscapes', split='train', mode='fine',
                           target_type=['instance', 'color', 'polygon'])

# For validation data
val_dataset = CustomCityscapes('./data/cityscapes', split='val', mode='fine',
                         target_type=['instance', 'color', 'polygon'])

# For test data
test_dataset = CustomCityscapes('./data/cityscapes', split='test', mode='fine',
                          target_type=['instance', 'color', 'polygon'])

# tensor, (inst, col, poly) = val_dataset[0]
tensor, bboxes = val_dataset[0]

v=0