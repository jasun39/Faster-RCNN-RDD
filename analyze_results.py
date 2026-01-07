import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

# Importy z Twojego projektu
from datasets import CustomDataset
from models.create_fasterrcnn_model import create_model
from utils.general import collate_fn

# Importy narzędzi COCO z torch_utils
from torch_utils.coco_utils import get_coco_api_from_dataset
from torch_utils.coco_eval import CocoEvaluator
from torch_utils import utils

def evaluate_and_get_coco_results(model, data_loader, device):
    """
    Zmodyfikowana funkcja ewaluacji, która zwraca obiekt CocoEvaluator,
    dzięki czemu mamy dostęp do surowych danych precision/recall.
    """
    model.eval()
    
    # Tworzenie API COCO dla datasetu
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"] # Interesują nas tylko bbox
    coco_evaluator = CocoEvaluator(coco, iou_types)

    print("Rozpoczynanie inferencji na zbiorze walidacyjnym...")
    
    cpu_device = torch.device("cpu")
    
    for images, targets in tqdm(data_loader, desc="Ewaluacja"):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        with torch.no_grad():
            outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    # Synchronizacja i akumulacja wyników
    print("Gromadzenie statystyk...")
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    return coco_evaluator

def main():
    # 1. Konfiguracja
    config_path = 'config/rdd.yaml'
    weights_path = 'runs/resnet_50_fpn/best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    val_dataset = CustomDataset.get_validation_subset(config, num_samples=1000)
    
    data_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # 3. Ładowanie modelu
    num_classes = config['dataset_params']['num_classes']
    print(f"Ładowanie modelu z {weights_path}...")
    
    build_model = create_model['fasterrcnn_resnet50_fpn']
    model = build_model(num_classes=num_classes, coco_model=False)
    
    checkpoint = torch.load(weights_path, map_location=device)
    # Obsługa zapisu z całego słownika lub tylko state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)

    # 4. Uruchomienie ewaluacji
    coco_evaluator = evaluate_and_get_coco_results(model, data_loader, device)
    
    # 5. Analiza wyników
    coco_eval = coco_evaluator.coco_eval['bbox']
    
    # Struktura eval['precision']: [TxRxKxAxM]
    # T: IoU thresholds (10: 0.5:0.05:0.95)
    # R: Recall thresholds (101: 0:0.01:1)
    # K: Classes (Categories)
    # A: Areas (4: all, small, medium, large) -> index 0 to 'all'
    # M: Max Detections (3: 1, 10, 100) -> index 2 to 100
    
    precision = coco_eval.eval['precision']
    
    class_mapping = config['test_params']['class_mapping']
    id_to_name = {v: k for k, v in class_mapping.items() if v != 0}
    cat_ids = coco_eval.params.catIds # ID kategorii faktycznie obecnych w ewaluacji
    
    results_data = []
    
    # Tablice do wykresu (dla IoU=0.5)
    pr_curves = {}
    
    print("\n--- Wyniki szczegółowe dla klas ---")
    
    for k_idx, cat_id in enumerate(cat_ids):
        class_name = id_to_name.get(cat_id, f"Class {cat_id}")
        
        # --- Obliczanie AP (Average Precision) ---
        # Średnia po wszystkich progach IoU (T), dla obszaru 'all' (A=0) i maxDets=100 (M=2)
        # To odpowiada metryce mAP @ 0.5:0.95 dla danej klasy
        p_k = precision[:, :, k_idx, 0, 2]
        
        # Ignorujemy wartości -1 (oznaczają brak detekcji/danych w tym obszarze)
        valid_mask = p_k > -1
        if valid_mask.sum() > 0:
            ap = np.mean(p_k[valid_mask])
        else:
            ap = 0.0
            
        results_data.append({
            "Class ID": cat_id,
            "Class Name": class_name,
            "AP (0.5:0.95)": ap
        })
        
        # --- Dane do wykresu PR (dla IoU=0.5) ---
        # T=0 odpowiada IoU=0.5
        pr_curve = precision[0, :, k_idx, 0, 2]
        pr_curves[class_name] = pr_curve

    # 6. Generowanie tabeli
    df = pd.DataFrame(results_data)
    print(df)
    df.to_csv("runs/ap_per_class.csv", index=False)
    print("Zapisano tabelę do runs/ap_per_class.csv")

    # 7. Rysowanie wykresu
    plt.figure(figsize=(10, 8))
    x_recall = np.linspace(0, 1, 101) # Standardowe 101 punktów recall COCO
    
    # Rysowanie linii dla każdej klasy
    valid_curves = []
    for class_name, curve in pr_curves.items():
        # Filtrujemy -1 (brak danych)
        valid_indices = curve > -1
        if valid_indices.sum() > 0:
            plt.plot(x_recall[valid_indices], curve[valid_indices], label=f"{class_name}", linewidth=2)
            valid_curves.append(curve)
    
    # Obliczanie "wspólnej" krzywej (Mean Precision)
    if valid_curves:
        # Zamieniamy listę arrayów na macierz, traktujemy -1 jako NaN do średniej
        all_curves = np.array(valid_curves)
        all_curves[all_curves == -1] = np.nan
        mean_curve = np.nanmean(all_curves, axis=0)
        
        # Rysowanie średniej
        plt.plot(x_recall, mean_curve, label="mAP (średnia)", color='black', linestyle='--', linewidth=3)

    #plt.title("Precision-Recall Curve (IoU=0.5)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    
    output_plot_path = "runs/pr_curve.png"
    plt.savefig(output_plot_path)
    print(f"Zapisano wykres do {output_plot_path}")
    plt.show()

if __name__ == '__main__':
    # Upewnij się, że folder runs istnieje
    os.makedirs('runs', exist_ok=True)
    main()