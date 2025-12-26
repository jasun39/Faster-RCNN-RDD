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
from torch.utils.tensorboard import SummaryWriter

from torch_utils.engine import train_one_epoch, evaluate
from utils.general import Averager, SaveBestModel, save_model, collate_fn
from utils.logging import csv_log
from models.create_fasterrcnn_model import create_model
from datasets import CustomDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Wczytanie konfiguracji z pliku YAML
    # Wczytanie konfiguracji
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_config = config['train_params']
    val_config = config['validation_params']
    dataset_config = config['dataset_params']

    print(f"Konfiguracja wczytana: {train_config['task_name']}")

    # Przygotowanie datasetu
    dataset = CustomDataset(
        root_path=dataset_config['root_path'],
        countries=dataset_config['countries'],
        class_mapping=dataset_config['class_mapping'],
        split='train'
    )

    # Podział na train/val (jeśli robisz to dynamicznie, a nie z plików folderów)
    # Sugeruję używać oddzielnych folderów jak w configu RDD, ale tu zostawiam Twoją logikę:
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # --- BLOK TRYBU DEBUG ---
    if args.debug:
        print("\n" + "="*40)
        print(" URUCHAMIANIE W TRYBIE DEBUG")
        print("="*40)

        # 1. Sprawdzenie poprawności tworzenia datasetu (Smoke Test)
        try:
            print("1. Weryfikacja próbki danych...")
            # Pobieramy 1. element (zwraca: img_tensor, target_dict, path)
            sample_img, sample_target = dataset[0]
            
            print(f"   - Kształt obrazu: {sample_img.shape}")
            print(f"   - Klucze targetu: {list(sample_target.keys())}")
            
            if 'boxes' in sample_target:
                print(f"   - Liczba ramek: {len(sample_target['boxes'])}")
                if len(sample_target['boxes']) > 0:
                    print(f"   - Przykładowa ramka: {sample_target['boxes'][0]}")
            
            print("   -> Dataset działa poprawnie.\n")
        except Exception as e:
            print(f"\n!!! BŁĄD PODCZAS TWORZENIA DATASETU: {e}")
            raise e

        # 2. Ograniczenie danych do 10 kroków
        # 10 kroków * batch_size = liczba potrzebnych próbek
        debug_samples = 1 * train_config['batch_size']
        
        # Upewniamy się, że nie żądamy więcej próbek niż mamy
        debug_samples_train = min(len(train_dataset), debug_samples)
        debug_samples_val = min(len(val_dataset), debug_samples)

        print(f"2. Przycinanie datasetów do 10 kroków ({train_config['batch_size']} img/batch)...")
        # Tworzymy podzbiory (Subset)
        train_dataset = torch.utils.data.Subset(train_dataset, range(debug_samples_train))
        val_dataset = torch.utils.data.Subset(val_dataset, range(debug_samples_val))
        
        print(f"   - Nowy rozmiar train: {len(train_dataset)} próbek")
        print(f"   - Nowy rozmiar val:   {len(val_dataset)} próbek")

        # 3. Wymuszenie 1 epoki (Early Stop po walidacji)
        args.epochs = 1
        print("3. Liczba epok ustawiona na 1 (Early Stop po zakończeniu walidacji).")
        print("="*40 + "\n")
    # ------------------------

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_config['batch_size'], 
        shuffle=False, 
        num_workers=2, 
        collate_fn=collate_fn
    )

    # Model
    build_model = create_model['fasterrcnn_resnet50_fpn']
    model = build_model(num_classes=dataset_config['num_classes'], pretrained=True)
    model.to(device)

    # Optymalizator
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=train_config['lr'], momentum=0.9, weight_decay=0.0005)

    # Scheduler (Warmup uruchomi się sam w 1 epoce wewnątrz train_one_epoch)
    # Tu definiujemy scheduler, który działa PO warmupie (redukcja LR w konkretnych epokach)
    lr_scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)

    writer = SummaryWriter(log_dir=os.path.join("runs", train_config['task_name']))
    
    # Klasa pomocnicza do uśredniania strat (z utils/general.py)
    train_loss_hist = Averager()
    train_loss_list = []
    loss_cls_list = []
    loss_box_reg_list = []
    loss_objectness_list = []
    loss_rpn_list = []
    train_loss_list_epoch = []
    val_map_05 = []
    val_map = []
    start_epochs = 0
    
    # Zmienne do śledzenia najlepszego modelu
    best_map = 0.0

    print("Rozpoczynam trening...")
    
    for epoch in range(start_epochs, train_config['num_epochs']):
        train_loss_hist.reset()

        # TRENING (z Warmupem wewnątrz)
        # Funkcja zwraca krotkę strat, ale aktualizuje też train_loss_hist
        metric_logger, \
        batch_loss_list, \
        batch_loss_cls_list, \
        batch_loss_box_reg_list, \
        batch_loss_objectness_list, \
        batch_loss_rpn_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            device, 
            epoch, 
            train_loss_hist,
            print_freq=50,
            scheduler=None # Możesz tu przekazać lr_scheduler jeśli używasz CosineAnnealing, dla MultiStepLR robimy step() ręcznie niżej
        )

        # Aktualizacja głównego schedulera
        lr_scheduler.step()

        # EWALUACJA (Obliczanie mAP)
        # Funkcja evaluate korzysta z pycocotools (poprzez torch_utils)
        # Zwraca stats: [AP50:95, AP50, AP75, AP_S, AP_M, AP_L, ...]
        coco_evaluator, _ = evaluate(model, val_loader, device=device)
        
        # Pobieranie wyników
        stats = coco_evaluator.coco_eval['bbox'].stats
        mAP = stats[0]    # mAP @ 0.5:0.95
        mAP50 = stats[1]  # mAP @ 0.5

        csv_log(
            log_dir=os.path.join("runs", train_config['task_name']), # lub inna ścieżka
            stats=stats,
            epoch=epoch,
            train_loss_list=[train_loss_hist.value], # Uśredniony loss
            loss_cls_list=[np.mean(batch_loss_cls_list)],
            loss_box_reg_list=[np.mean(batch_loss_box_reg_list)],
            loss_objectness_list=[np.mean(batch_loss_objectness_list)],
            loss_rpn_list=[np.mean(batch_loss_rpn_list)]
        )

        print(f"Epoka {epoch+1} zakończona. mAP: {mAP:.4f}, mAP50: {mAP50:.4f}, Loss: {train_loss_hist.value:.4f}")

        # Logowanie do Tensorboard
        writer.add_scalar('Training/Loss', train_loss_hist.value, epoch)
        writer.add_scalar('Validation/mAP', mAP, epoch)
        writer.add_scalar('Validation/mAP_50', mAP50, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # ZAPISYWANIE MODELU
        
        # Zapisz najlepszy model
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), os.path.join(train_config['task_name'], 'best_model.pth'))
            print(f"Zapisano nowy najlepszy model! (mAP: {best_map:.4f})")

        # Zapisz ostatni model (do wznawiania)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_map': best_map,
        }, os.path.join(train_config['task_name'], 'last_model.pth'))

    writer.close()
    print('Trening zakończony.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trening sieci Faster R-CNN')
    parser.add_argument('--config', dest='config_path',
                        default='config/rdd.yaml', type=str,
                        help='Ścieżka do pliku konfiguracyjnego YAML')

    parser.add_argument('--debug', action='store_true', help='Uruchom krótki trening testowy (Dry Run)')
    args = parser.parse_args()
    train(args)