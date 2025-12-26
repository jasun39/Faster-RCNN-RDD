import os
import torch
import xml.etree.ElementTree as ET
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class CustomDataset(Dataset):
    r"""
    Klasa obsługująca zbiór RDD2022. Automatycznie przeszukuje foldery krajów 
    w poszukiwaniu zdjęć i adnotacji XML.
    """
    def __init__(self, root_path, countries, class_mapping, split='train'):
        self.class_mapping = class_mapping
        self.split = split
        self.images_data = []

        folder_name = 'test' if split == 'test' else 'train'

        # Zbieranie ścieżek do wszystkich zdjęć ze wszystkich wybranych krajów
        for country in countries:
            # Określamy folder źródłowy: dla train/val zawsze bierzemy z 'train'
            skip_count = 0
            img_dir = os.path.join(root_path, country, folder_name, 'images')
            ann_dir = os.path.join(root_path, country, folder_name, 'annotations', 'xmls')
            
            if not os.path.exists(img_dir):
                skip_count += 1
                continue
            
            for img_name in os.listdir(img_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(img_dir, img_name)
                    ann_path = os.path.join(ann_dir, img_name.replace('.jpg', '.xml'))

                    # Dodaj tylko jeśli istnieją oba pliki (zdjęcie i adnotacja)
                    if split == 'test':
                        self.images_data.append({'img_path': img_path, 'ann_path': None})
                    elif split == 'train' and os.path.exists(ann_path):
                        self.images_data.append({'img_path': img_path, 'ann_path': ann_path})
            
            #print(f"Pominięto {skip_count} w zbiorze {country}")

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        data = self.images_data[idx]
        img = Image.open(data['img_path']).convert("RGB")
        img_tensor = F.to_tensor(img)
        
        if self.split == 'test' or data['ann_path'] is None:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            }
            return img_tensor, target

        # Parsowanie pliku XML z adnotacjami
        tree = ET.parse(data['ann_path'])
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            label_name = obj.find('name').text
            if label_name not in self.class_mapping:
                continue
                
            # Pobieranie współrzędnych ramki (xmin, ymin, xmax, ymax)
            xmlbox = obj.find('bndbox')
            bbox = [
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymax').text)
            ]
            
            boxes.append(bbox)
            labels.append(self.class_mapping[label_name])
            
        # Przygotowanie słownika target zgodnie z wymaganiami pętli treningowej
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        # Jeśli obraz nie ma żadnych ramek, dodajemy ramkę "pustą" (wymagane technicznie)
        if len(boxes) == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target