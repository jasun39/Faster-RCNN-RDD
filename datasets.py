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

    def get_annotations(self, index):
        data = self.images_data[index]
        
        # Szybki odczyt rozmiaru bez dekodowania pikseli
        with Image.open(data['img_path']) as img:
            width, height = img.size
        
        boxes = []
        labels = []
        
        if data['ann_path'] and os.path.exists(data['ann_path']):
            tree = ET.parse(data['ann_path'])
            root = tree.getroot()
            
            for obj in root.findall('object'):
                label_name = obj.find('name').text
                if label_name not in self.class_mapping:
                    continue
                    
                xmlbox = obj.find('bndbox')
                bbox = [
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymax').text)
                ]
                boxes.append(bbox)
                labels.append(self.class_mapping[label_name])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        
        if len(boxes) > 0:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)
        else:
            area = torch.as_tensor([], dtype=torch.float32)
            iscrowd = torch.as_tensor([], dtype=torch.int64)
            boxes_tensor = boxes_tensor.reshape(-1, 4)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([index]),
            'area': area,
            'iscrowd': iscrowd
        }
        
        # Zwracamy target ORAZ wymiary
        return target, height, width

    def __getitem__(self, index):
        data = self.images_data[index]
        img = Image.open(data['img_path']).convert("RGB")
        img_tensor = F.to_tensor(img)

        boxes = []
        labels = []
        
        # Sprawdzenie czy plik adnotacji istnieje
        if os.path.exists(data['ann_path']):
            tree = ET.parse(data['ann_path'])
            root = tree.getroot()
            
            for obj in root.findall('object'):
                label_name = obj.find('name').text
                if label_name not in self.class_mapping:
                    continue
                    
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                ymin = float(xmlbox.find('ymin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymax = float(xmlbox.find('ymax').text)

                if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
                    print(f"Ostrzeżenie: Ignorowanie błędnej ramki w {data['img_path']} -> {xmin, ymin, xmax, ymax}")
                    continue

                bbox = [xmin, ymin, xmax, ymax]
                boxes.append(bbox)
                labels.append(self.class_mapping[label_name])

        # Konwersja na Tensory i obliczanie Area/Iscrowd (wg wzorca)
        
        # Najpierw konwersja list na tensory
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        
        # Sprawdzenie długości (czy są jakiekolwiek obiekty)
        if len(boxes) > 0:
            # Obliczanie pola powierzchni: (xmax-xmin) * (ymax-ymin)
            # Indeksy: 0=xmin, 1=ymin, 2=xmax, 3=ymax
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)
        else:
            # Obsługa pustych ramek - puste tensory o odpowiednich typach
            area = torch.as_tensor([], dtype=torch.float32)
            iscrowd = torch.as_tensor([], dtype=torch.int64)
            # Wymuszamy kształt (0, 4) dla pustego tensora ramek, żeby model nie zgłaszał błędu
            boxes_tensor = boxes_tensor.reshape(-1, 4)

        # Przygotowanie słownika target
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([index]), # Wymagane przez COCO eval
            'area': area,
            'iscrowd': iscrowd
        }

        return img_tensor, target