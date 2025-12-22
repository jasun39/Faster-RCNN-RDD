import torch
import torch.nn as nn
import torchvision
import math
from torchvision.models import resnet18, ResNet18_Weights
from .rpn import RegionProposalNetwork
from .roi_head import ROIHead
from .utils import transform_boxes_to_original_size

class FasterRCNN(nn.Module):
    def __init__(self, model_config, num_classes):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
            )
        self.rpn = RegionProposalNetwork(model_config['backbone_out_channels'],
                                         scales=model_config['scales'],
                                         aspect_ratios=model_config['aspect_ratios'],
                                         model_config=model_config
                                         )
        self.roi_head = ROIHead(model_config, 
                                num_classes, 
                                in_channels=model_config['backbone_out_channels']
                                )
        for layer in [self.backbone[0], self.backbone[1], self.backbone[4]]: # conv1, bn1, layer1
            for p in layer.parameters():
                p.requires_grad = False
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']
    
    def normalize_resize_image_and_boxes(self, image, bboxes=None):
        dtype, device = image.dtype, image.device
        
        # Normalizacja
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        
        # Przeskalowanie
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(
            float(self.min_size) / min_size, 
            float(self.max_size) / max_size)
        scale_factor = scale.item()
        
        image = torch.nn.functional.interpolate(
            image[None], 
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )[0]

        if bboxes is not None and bboxes.numel() > 0:
            new_h, new_w = image.shape[-2:]
            ratios_h = new_h / h
            ratios_w = new_w / w
            
            xmin, ymin, xmax, ymax = bboxes.unbind(1)
            
            xmin = xmin * ratios_w
            xmax = xmax * ratios_w
            ymin = ymin * ratios_h
            ymax = ymax * ratios_h
            
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
            
        return image, bboxes
    
    def batch_images(self, images, size_divisible=32):
        """
        Skleja listę tensorów w jeden Batch Tensor z paddingiem (zera).
        Wymagane, bo ResNet potrzebuje jednego tensora 4D.
        """
        # Znajdź max H i W w całym batchu
        max_size = list(max(s) for s in zip(*[img.shape for img in images]))
        
        # Wyrównanie do wielokrotności 32 (wymagane przez stride w ResNet)
        stride = float(size_divisible)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)

        batch_shape = [len(images)] + max_size
        
        # Pusty tensor wypełniony zerami (padding)
        batched_imgs = images[0].new_full(batch_shape, 0)
        
        for img, pad_img in zip(images, batched_imgs):
            # Kopiujemy obraz w lewy górny róg
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs
    
    def forward(self, images, targets=None):
        # Przechowujemy oryginalne rozmiary do odwrócenia skalowania na końcu
        original_image_sizes = []
        for img in images:
            old_shape = img.shape[-2:]
            original_image_sizes.append((old_shape[0], old_shape[1]))
        
        images_res = []
        if self.training:
            for i, img in enumerate(images):
                target_boxes = targets[i]['bboxes']
                new_img, new_boxes = self.normalize_resize_image_and_boxes(img, target_boxes)
                images_res.append(new_img)
                targets[i]['bboxes'] = new_boxes
        else:
            # Walidacja/Test
            for img in images:
                new_img, _ = self.normalize_resize_image_and_boxes(img, None)
                images_res.append(new_img)
        
        image_tensor = self.batch_images(images_res)

        # Backbone
        feat = self.backbone(image_tensor)

        # RPN
        image_shapes = [img.shape[-2:] for img in images_res]
        
        rpn_output = self.rpn(image_tensor, feat, image_shapes, targets)
        proposals = rpn_output['proposals']
        
        # ROI Head
        frcnn_output = self.roi_head(feat, proposals,image_shapes, targets)
        if not self.training:
            final_boxes = []
            for i, box in enumerate(frcnn_output['boxes']):
                rescaled_box = transform_boxes_to_original_size(
                    box, 
                    image_shapes[i], 
                    original_image_sizes[i]
                )
                final_boxes.append(rescaled_box)
            frcnn_output['boxes'] = final_boxes

        return rpn_output, frcnn_output