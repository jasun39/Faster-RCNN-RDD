import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import argparse
import numpy as np
import yaml
import random
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model.faster_rcnn import FasterRCNN
from dataset.voc import VOCDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, dataloader, device, debug_mode=False):
    r"""
    Przeprowadza walidację na zbiorze walidacyjnym i oblicza metrykę mAP.
    Przenosi dane na CPU, aby uniknąć błędów pamięci VRAM podczas akumulacji wyników.
    """
    model.eval()
    metric = MeanAveragePrecision()
    metric_updated = False
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Walidacja mAP") as pbar:
            for step, (im, target, fname) in enumerate(pbar):            
                if debug_mode and step >= 5:
                    early_stop_triggered = True
                    break
            
            im = im.float().to(device)
            # Uzyskanie predykcji (w trybie eval model zwraca przefiltrowane ramki)
            rpn_output, frcnn_output = model(im)
            
            # Przygotowanie wyników do formatu torchmetrics
            preds = [
                dict(
                    boxes=frcnn_output['boxes'].to('cpu'),
                    scores=frcnn_output['scores'].to('cpu'),
                    labels=frcnn_output['labels'].to('cpu'),
                )
            ]
            
            # Przygotowanie celów (Ground Truth)
            targets = [
                dict(
                    boxes=target['bboxes'][0].to('cpu'),
                    labels=target['labels'][0].to('cpu'),
                )
            ]
            
            # Aktualizacja metryki, jeśli zdjęcie zawiera jakiekolwiek obiekty
            if targets[0]['boxes'].shape[0] > 0:
                metric.update(preds, targets)
                metric_updated = True

    if early_stop_triggered:
        print("DEBUG: Przerwanie walidacji po 5 krokach.")

    if not metric_updated:
        if debug_mode:
             print("DEBUG: Wylosowane próbki walidacyjne nie zawierały obiektów.")
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
        'train', 
        root_path=dataset_config['root_path'], 
        countries=dataset_config['countries'], 
        class_mapping=dataset_config['class_mapping'],
        ratio = train_config['train_val_ratio'],
        seed=seed
    )
    val_dataset = VOCDataset(
        'val', 
        root_path=dataset_config['root_path'], 
        countries=dataset_config['countries'], 
        class_mapping=dataset_config['class_mapping'],
        ratio = train_config['train_val_ratio'],
        seed=seed
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Inicjalizacja modelu Faster R-CNN
    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    faster_rcnn_model.to(device)

    # Przygotowanie folderu na wyniki treningu
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
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
                # --- DEBUG MODE: Przerwij pętlę treningową po 10 krokach ---
                if args.debug and step >= 10:
                    early_stop_triggered = True
                    break
                im = im.float().to(device)
                target['bboxes'] = target['bboxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
                
                # Przejście przez model i obliczenie strat
                rpn_output, frcnn_output = faster_rcnn_model(im, target)
                
                rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
                frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
                loss = rpn_loss + frcnn_loss
                
                # Zapisywanie danych do statystyk
                rpn_cls_losses.append(rpn_output['rpn_classification_loss'].item())
                rpn_loc_losses.append(rpn_output['rpn_localization_loss'].item())
                frcnn_cls_losses.append(frcnn_output['frcnn_classification_loss'].item())
                frcnn_loc_losses.append(frcnn_output['frcnn_localization_loss'].item())
                
                # Akumulacja gradientów
                (loss / acc_steps).backward()
                
                if (step + 1) % acc_steps == 0:
                    optimizer.step()
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

    #parser.add_argument('--debug', action='store_true', help='Uruchom krótki trening testowy (Dry Run)')
    args = parser.parse_args()
    train(args)