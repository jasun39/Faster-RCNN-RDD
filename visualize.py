import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import nms
from datasets import CustomDataset
from models.create_fasterrcnn_model import create_model


model = create_model['fasterrcnn_resnet18']

def visualize(args):
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_config = config['test_params']

    id_to_class = {v: k for k, v in test_config['class_mapping'].items()}

    test_dataset = CustomDataset(
        root_path=test_config['root_path'],
        countries=test_config['countries'],
        class_mapping=test_config['class_mapping'],
        split='test'
    )

    checkpoint = torch.load(test_config['weights_path'], map_location=device)
    build_model = create_model['fasterrcnn_resnet18']
    model = build_model(num_classes=test_config['num_classes'], coco_model=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    detection_threshold = test_config['threshold']
    nms_threshold = test_config.get('nms_threshold', 0.2)

    pred_boxes = {}
    box_id = 1
    frame_count = 0
    total_fps = 0
    
    COLORS = {
        'PPD': 'cyan',
        'PPP': 'green',
        'PS' : 'yellow',
        'WY' : 'red'
        }

    for i in range(args.num_images):
        if i >= len(test_dataset):
            break
            
        # Pobranie danych (zwraca img_tensor i target)
        img_tensor, _ = test_dataset[i]
        
        # Przygotowanie do modelu
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(input_tensor)[0]

        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']

        # Filtrowanie wyników
        mask = scores > detection_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        keep_indices = nms(boxes, scores, nms_threshold)
        
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

        # Pobieranie nazw klas dla wykrytych obiektów
        current_labels = []
        current_colors = []

        for label, score in zip(labels, scores):
            label_text = f"{id_to_class.get(label.item(), '--')} : {score:.2f}"
            current_labels.append(label_text)
            
            color = COLORS.get(id_to_class.get(label.item()), 'white')
            current_colors.append(color)
        
        if len(boxes) > 0:
            # Konwersja obrazu na uint8 dla draw_bounding_boxes
            img_uint8 = (img_tensor * 255).to(torch.uint8)
            # Rysowanie ramek
            result_img = draw_bounding_boxes(
                img_uint8, 
                boxes=boxes, 
                labels=current_labels, 
                colors=current_colors, 
                width=2,
                label_colors='black',
                fill_labels=True
            )
        else:
            result_img = img_uint8

        # Wyświetlanie Matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(to_pil_image(result_img))
        plt.title(f"Image {i} - Predictions (threshold > {detection_threshold})")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test sieci Faster R-CNN')
    parser.add_argument('--config', dest='config_path',
                        default='config/rdd.yaml', type=str,
                        help='Ścieżka do pliku konfiguracyjnego YAML')
    parser.add_argument('--num_images', dest='num_images', default=20, type=int)
    args = parser.parse_args()
    visualize(args)