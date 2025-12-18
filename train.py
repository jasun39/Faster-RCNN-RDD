import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm

from model import FasterRCNN 
from dataset import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Ładowanie konfiguracji YAML
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Ustawianie powtarzalności wyników
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Przygotowanie Datasetu i Loadera
    # Używamy formatu VOC
    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'])
    train_dataset = DataLoader(voc,
                               batch_size=1, # Ze względu na implementacje Faster R-CNN 
                               shuffle=True,
                               num_workers=4)
    
    # Inicjalizacja modelu
    faster_rcnn_model = FasterRCNN(model_config,
                                   num_classes=dataset_config['num_classes'])
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    # Konfiguracja optymalizatora i harmonogramu LR
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              faster_rcnn_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    # Główna pętla treningowa
    for i in range(num_epochs):
        rpn_cls_losses, rpn_loc_losses = [], []
        frcnn_cls_losses, frcnn_loc_losses = [], []
        optimizer.zero_grad()
        
        for im, target, fname in tqdm(train_dataset, desc=f"Epoch {i+1}/{num_epochs}"):
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            
            # Przekazanie danych z modelu
            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            
            # Sumowanie strat z RPN i modelu
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            loss = rpn_loss + frcnn_loss
            
            # Zapisanie danych strat
            rpn_cls_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_loc_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_cls_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_loc_losses.append(frcnn_output['frcnn_localization_loss'].item())
            
            # Akumulacja gradientów
            loss = loss / acc_steps
            loss.backward()
            
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1
            
        # Zapisywanie punktu kontrolnego po kadej epoce (checkpoint)
        torch.save(faster_rcnn_model.state_dict(), 
                   os.path.join(train_config['task_name'], train_config['ckpt_name']))
        
        # Wyświetlanie średnich strat dla epoki
        print(f'\nZakończono epokę {i+1}')
        print(f'Strata klasyfikacji RPN {np.mean(rpn_cls_losses):.4f} | Strata lokalizacji RPN : {np.mean(rpn_loc_losses):.4f}')
        print(f'Strata klasyfikacji F-RCNN : {np.mean(frcnn_cls_losses):.4f} | Strata lokalizacji F-RCNN : {np.mean(frcnn_loc_losses):.4f}')
        
        scheduler.step()
        
    print('Trening zakończony...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argumenty do trenowania sieci Faster R-CNN')
    parser.add_argument('--config', dest='config_path',
                        default='config/rdd.yaml', type=str)
    args = parser.parse_args()
    train(args)