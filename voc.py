import os
import torch
import xml.etree.ElementTree as ET
import random
from PIL import Image
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    r"""
    Klasa obsługująca zbiór RDD2022. Automatycznie przeszukuje foldery krajów 
    w poszukiwaniu zdjęć i adnotacji XML.
    """
    def __init__(self, split, root_path, countries, class_mapping, ratio, seed=1111):
        self.split = split  # 'train', 'val' lub 'test'
        self.class_mapping = class_mapping
        self.images_data = []
        self.ratio = ratio
        self.seed = seed
        all_train_data = []

        # Zbieranie ścieżek do wszystkich zdjęć ze wszystkich wybranych krajów
        for country in countries:
            # Określamy folder źródłowy: dla train/val zawsze bierzemy z 'train'
            skip_count = 0
            current_split = 'test' if split == 'test' else 'train'
            img_dir = os.path.join(root_path, country, current_split, 'images')
            ann_dir = os.path.join(root_path, country, current_split, 'annotations', 'xmls')
            
            if not os.path.exists(img_dir):
                skip_count += 1
                continue
            
            for img_name in os.listdir(img_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(img_dir, img_name)
                    ann_path = os.path.join(ann_dir, img_name.replace('.jpg', '.xml'))

                    # Dodaj tylko jeśli istnieją oba pliki (zdjęcie i adnotacja)
                    if current_split == 'test' or os.path.exists(ann_path):
                        item = {'img_path': img_path, 'ann_path': ann_path}
                        
                        if current_split == 'test':
                            self.images_data.append(item)
                        else:
                            # Dla train/val wrzucamy do wspólnego all_train_data
                            all_train_data.append(item)
            
            print(f"Pominięto {skip_count} w zbiorze {country}")
                        
        # Logika podziału train/val
        if split in ['train', 'val']:
            random.seed(seed)
            random.shuffle(all_train_data)
            val_size = int(len(all_train_data) * ratio)
            
            if split == 'val':
                self.images_data = all_train_data[:val_size]
            else:
                self.images_data = all_train_data[val_size:]

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        info = self.images_data[idx]
        img = Image.open(info['img_path']).convert("RGB")
        
        # Przekształcenie obrazu na tensor (zgodnie z wymaganiami modelu)
        img_tensor = torch.tensor(list(img.getdata())).reshape(img.size[1], img.size[0], 3)
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0 # (C, H, W)
        
        if self.split == 'test':
            return img_tensor, info['img_path']

        # Parsowanie pliku XML z adnotacjami
        tree = ET.parse(info['ann_path'])
        root = tree.getroot()
        
        bboxes = []
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
            
            bboxes.append(bbox)
            labels.append(self.class_mapping[label_name])
            
        # Przygotowanie słownika target zgodnie z wymaganiami pętli treningowej
        target = {
            'bboxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        # Jeśli obraz nie ma żadnych ramek, dodajemy ramkę "pustą" (wymagane technicznie)
        if len(bboxes) == 0:
            target['bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target, info['img_path']