import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import argparse
import numpy as np
import yaml
import random
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model.faster_rcnn import FasterRCNN
from voc import VOCDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    return tuple(zip(*batch))

def validate(model, dataloader, device, debug_mode=False):
    r"""
    Przeprowadza walidację na zbiorze walidacyjnym i oblicza metrykę mAP.
    Przenosi dane na CPU, aby uniknąć błędów pamięci VRAM podczas akumulacji wyników.
    """
    model.eval()
    metric = MeanAveragePrecision()
    metric_updated = False
    early_stop_triggered = False
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Walidacja mAP") as pbar:
            for step, (im, target, fname) in enumerate(pbar):            
                if debug_mode and step >= 5:
                    early_stop_triggered = True
                    break

                images = list(image.float().to(device) for image in im)
                
                # Uzyskanie predykcji
                with torch.amp.autocast('cuda'):
                    _, frcnn_output = model(images)
                
                preds = []
                targets_list = []
                
                # frcnn_output['boxes'] jest listą o długości batch_size
                batch_size_current = len(frcnn_output['boxes'])
                
                for i in range(batch_size_current):
                    # Przygotowanie predykcji dla i-tego zdjęcia
                    preds.append(dict(
                        boxes=frcnn_output['boxes'][i].to('cpu'),
                        scores=frcnn_output['scores'][i].to('cpu'),
                        labels=frcnn_output['labels'][i].to('cpu'),
                    ))
                    
                    # Przygotowanie targetu dla i-tego zdjęcia
                    targets_list.append(dict(
                        boxes=target[i]['bboxes'].to('cpu'),
                        labels=target[i]['labels'].to('cpu'),
                    ))

                # Aktualizacja metryki (torchmetrics obsługuje listy słowników)
                # Sprawdzamy czy są jakiekolwiek obiekty w targetach w całym batchu
                has_objects = any(t['boxes'].shape[0] > 0 for t in targets_list)
                
                if has_objects:
                    metric.update(preds, targets_list)
                    metric_updated = True

    
    if debug_mode and not metric_updated:
         print("DEBUG: Wylosowane próbki walidacyjne nie zawierały obiektów.")
         return 0.0
    
    if not metric_updated:
        return 0.0

    result = metric.compute()
    return result['map_50'].item() # Zwracamy mAP dla progu IoU=0.5

def train(args):
    # Wczytanie konfiguracji z pliku YAML
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Błąd wczytywania YAML: {exc}")
            return
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # --- DEBUG MODE: Nadpisanie parametrów ---
    if args.debug:
        print("-" * 40)
        print("URUCHOMIONO TRYB DEBUGOWANIA")
        print("Trening zostanie skrócony do minimum, aby przetestować przepływ danych.")
        print("-" * 40)
        train_config['num_epochs'] = 1
        train_config['ckpt_name'] = 'debug_checkpoint.pth'
        # Zmniejszamy patience, żeby early stopping nie blokował (choć przy 1 epoce to bez znaczenia)
        train_config['early_stopping_patience'] = 1 
    # -----------------------------------------
    
    # Ustawianie powtarzalności wyników
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Tworzenie DataLoaderów dla treningu i walidacji z automatycznym podziałem
    train_dataset = VOCDataset( 
        root_path=dataset_config['root_path'], 
        countries=dataset_config['countries'], 
        class_mapping=dataset_config['class_mapping'],
        split='train'
    )
    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(train_dataset, [train_config['train_val_ratio'], 1-train_config['train_val_ratio']], generator=generator)

    train_loader = DataLoader(
        train_subset, 
        batch_size=train_config['batch_size'], 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=train_config['batch_size'], 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Inicjalizacja modelu Faster R-CNN
    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    faster_rcnn_model.to(device)

    # Przygotowanie folderu na wyniki treningu
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    #Inicjalizacja Scalera do Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    # Konfiguracja optymalizatora i harmonogramu uczenia
    optimizer = torch.optim.SGD(
        lr=train_config['lr'],
        params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        weight_decay=5E-4,
        momentum=0.9
    )
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
    # Parametry Early Stopping
    patience = train_config.get('early_stopping_patience', 5)
    best_map = 0.0
    epochs_without_improvement = 0
    num_epochs = train_config['num_epochs']
    acc_steps = train_config['acc_steps']

    # Główna pętla treningowa
    for i in range(num_epochs):
        faster_rcnn_model.train()
        rpn_cls_losses, rpn_loc_losses = [], []
        frcnn_cls_losses, frcnn_loc_losses = [], []
        optimizer.zero_grad()
        early_stop_triggered = False
        
        with tqdm(train_loader, desc=f"Epoka {i+1}/{num_epochs}") as pbar:
            for step, (im, target, fname) in enumerate(pbar):            
                if args.debug and step >= 10:
                    early_stop_triggered = True
                    break
                
                images = list(image.float().to(device) for image in im)
                
                targets = []
                for t in target:
                    d = {}
                    for k, v in t.items():
                        if k == 'labels':
                            d[k] = v.long().to(device)
                        else:
                            d[k] = v.float().to(device)
                    targets.append(d)
                
                with torch.amp.autocast('cuda'):
                    rpn_output, frcnn_output = faster_rcnn_model(images, targets)
                    
                    rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
                    frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
                    loss = rpn_loss + frcnn_loss
                    
                    loss = loss / acc_steps
                
                rpn_cls_losses.append(rpn_output['rpn_classification_loss'].item())
                rpn_loc_losses.append(rpn_output['rpn_localization_loss'].item())
                frcnn_cls_losses.append(frcnn_output['frcnn_classification_loss'].item())
                frcnn_loc_losses.append(frcnn_output['frcnn_localization_loss'].item())
                
                scaler.scale(loss).backward()
                
                if (step + 1) % acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
        
        if early_stop_triggered:
            print(f"DEBUG: Przerwanie epoki treningowej po 10 krokach.")
        
        # Wyświetlanie raportu strat po epoce
        print(f'Zakończono epokę {i+1}')
        print(f'Strata RPN: Klasyfikacja {np.mean(rpn_cls_losses):.4f}, Lokalizacja {np.mean(rpn_loc_losses):.4f}')
        print(f'Strata F-RCNN: Klasyfikacja {np.mean(frcnn_cls_losses):.4f}, Lokalizacja {np.mean(frcnn_loc_losses):.4f}')
        
        # Etap walidacji (mAP)
        print("Rozpoczynanie walidacji...")
        current_map = validate(faster_rcnn_model, val_loader, device, debug_mode=args.debug)
        print(f'mAP@50 na zbiorze walidacyjnym: {current_map:.4f}')

        # Sprawdzenie warunków wczesnego zatrzymania
        if current_map > best_map:
            best_map = current_map
            epochs_without_improvement = 0
            # Zapisywanie najlepszej wersji modelu
            torch.save(faster_rcnn_model.state_dict(), 
                       os.path.join(train_config['task_name'], 'best_' + train_config['ckpt_name']))
            print(f"Nowe najlepsze mAP! Zapisano model: best_{train_config['ckpt_name']}")
        else:
            epochs_without_improvement += 1
            print(f"Brak poprawy mAP. Próba {epochs_without_improvement} z {patience}")
            
        # Zapisywanie ostatniego checkpointu (na wypadek wznowienia treningu)
        torch.save(faster_rcnn_model.state_dict(), 
                   os.path.join(train_config['task_name'], train_config['ckpt_name']))
        
        # Przerwanie treningu, jeśli model przestał robić postępy
        if epochs_without_improvement >= patience:
            print(f"Uruchomiono wczesne zatrzymanie. Trening przerwany po {i+1} epokach.")
            break

        scheduler.step()
        
    print('\nProces treningowy zakończony.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trening sieci Faster R-CNN')
    parser.add_argument('--config', dest='config_path',
                        default='config/rdd.yaml', type=str,
                        help='Ścieżka do pliku konfiguracyjnego YAML')

    parser.add_argument('--debug', action='store_true', help='Uruchom krótki trening testowy (Dry Run)')
    args = parser.parse_args()
    train(args)