from torchvision.datasets import Cityscapes
from torchvision import transforms
import torch
# Define a transformation to convert PIL images to PyTorch Tensors
transform = transforms.ToTensor()

class CustomCityscapes(Cityscapes):

    def __getitem__(self, index):

        # Get data using the original __getitem__ from Cityscapes
        img, targets = super().__getitem__(index)
        
        # Convert the PIL Image to a PyTorch Tensor
        img_tensor = transform(img)

        # Process polygons to compute bounding boxes
        if 'polygon' in self.target_type:
            poly_index = self.target_type.index('polygon')
            poly = targets[poly_index]
            poly_list_with_bboxes, bboxes = self.polygons_to_bboxes(poly)
            targets[poly_index]['objects'] = poly_list_with_bboxes
            bboxes = torch.as_tensor(bboxes, dtype=torch.int)
        # Return the tensor image and the targets
        return img_tensor, poly_list_with_bboxes

    def polygons_to_bboxes(self, polygons):

        new_poly = []
        bboxes = []
        for polygon in polygons['objects']:

            # Calculate bounding boxes from polygons
            polygon_points = polygon['polygon']
            min_x = min(point[0] for point in polygon_points)
            max_x = max(point[0] for point in polygon_points)
            min_y = min(point[1] for point in polygon_points)
            max_y = max(point[1] for point in polygon_points)
            bbox = [min_x, min_y, max_x, max_y]
            if polygon['label'] == 'out of roi' or polygon['label'] == 'sky':
                v=0
                continue
            if min_x == 0 and min_y == 0 and max_x == 2048 and max_y == 1024:
                v=0
                continue
            # Create a dictionary with the polygon, label, and bounding box
            curr_dict = {}
            curr_dict['label'] = polygon['label']
            curr_dict['polygon'] = polygon_points
            curr_dict['bbox'] = bbox
            new_poly.append(curr_dict)
            bboxes.append(bbox)

        return new_poly, bboxes

# For training data
train_dataset = CustomCityscapes('../data/cityscapes', split='train', mode='fine',
                           target_type=['instance', 'color', 'polygon'])

# For validation data
val_dataset = CustomCityscapes('../data/cityscapes', split='val', mode='fine',
                         target_type=['instance', 'color', 'polygon'])

# For test data
test_dataset = CustomCityscapes('../data/cityscapes', split='test', mode='fine',
                          target_type=['instance', 'color', 'polygon'])

# tensor, (inst, col, poly) = val_dataset[0]
tensor, bboxes = val_dataset[0]

v=0