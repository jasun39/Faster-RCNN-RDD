import os
import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from tqdm import tqdm
from torch.utils.data import DataLoader

# Importy z Twojego projektu
from models.create_fasterrcnn_model import create_model
from datasets import CustomDataset
from utils.general import collate_fn
from torch_utils.engine import evaluate


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    class_mapping = config['test_config']['class_mapping']
    subset = CustomDataset.get_validation_subset(config, num_samples=1000)
    
    data_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Ustalanie liczby klas (z mapowania w configu)
    num_classes = config['dataset_params']['num_classes'] 
    
    checkpoint = torch.load(test_config['weights_path'], map_location=device)
    build_model = create_model['fasterrcnn_resnet50_fpn']
    model = build_model(num_classes=test_config['num_classes'], coco_model=False)
    model.load_state_dict(checkpoint)
    model.to(device)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ewaluacja modelu RDD (AP, AR, FPS, Confusion Matrix)')
    parser.add_argument('--config', dest='config_path', default='config/rdd.yaml', type=str,
                        help='Ścieżka do pliku konfiguracyjnego YAML')
    parser.add_argument('--weights', dest='weights_path', default='runs/resnet_50_fpn/best_model.pth', type=str,
                        help='Ścieżka do wytrenowanego modelu (.pth)')
    parser.add_argument('--batch_size', default=8, type=int, help='Rozmiar batcha')
    
    args = parser.parse_args()
    
    # Tworzenie folderu na wyniki jeśli nie istnieje
    os.makedirs('runs', exist_ok=True)
    
    main(args)