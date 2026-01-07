import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def generate_and_plot_pr_curve(gt_path, predictions_path, output_image='pr_curve.png'):
    """
    Rysuje zbiorczy wykres Precision-Recall tylko dla wybranych klas (1, 2, 3, 4).
    """
    print(f"--- Generowanie wykresu PR (Per-Class) ---")
    
    # Definiujemy akceptowalne klasy
    valid_classes = {1, 2, 3, 4}

    # 1. Wczytanie danych
    coco_gt = COCO(gt_path)
    try:
        coco_dt = coco_gt.loadRes(predictions_path)
    except Exception as e:
        print(f"Błąd wczytywania predykcji: {e}")
        return

    # 2. Ewaluacja
    # Uwaga: COCOeval domyślnie liczy dla wszystkich klas w GT.
    # Można by użyć coco_eval.params.catIds = [1, 2, 3, 4], ale
    # zrobimy filtrowanie przy rysowaniu, aby nie psuć wewnętrznej logiki ewaluatora.
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 3. Przygotowanie danych do wykresu
    # Tablica precision: [TxRxKxAxM] -> [10 (IoU), 101 (Recall), K (Klasy), 4 (Area), 3 (MaxDets)]
    precision_array = coco_eval.eval['precision'][0, :, :, 0, 2]
    
    x_recall = np.linspace(0, 1, 101)
    
    # Pobieramy identyfikatory klas
    cat_ids = coco_eval.params.catIds

    plt.figure(figsize=(10, 8))

    # Zdefiniujmy paletę kolorów
    colors = plt.cm.get_cmap('tab10', len(cat_ids))
    
    # Lista do przechowywania indeksów klas, które są akceptowalne (do obliczenia średniej)
    valid_indices_for_mean = []

    # A. Rysowanie linii dla poszczególnych klas
    for i, cat_id in enumerate(cat_ids):
        # --- FILTROWANIE ---
        # Jeśli klasa nie jest w zbiorze {1, 2, 3, 4}, pomijamy ją
        if cat_id not in valid_classes:
            continue

        # Dodajemy indeks do listy, aby później policzyć średnią tylko z tych klas
        valid_indices_for_mean.append(i)

        # Pobieramy nazwę klasy z Ground Truth
        cat_info = coco_gt.loadCats(cat_id)[0]
        cat_name = cat_info['name']
        
        # Dane dla konkretnej klasy
        class_precision = precision_array[:, i]
        
        # Rysujemy tylko, jeśli są jakieś dane (średnia > -1)
        if np.mean(class_precision) > -1:
            plt.plot(x_recall, class_precision, label=f'{cat_name} (ID: {cat_id})', 
                     linewidth=2, alpha=0.8, color=colors(i))

    # B. Rysowanie głównej linii średniej (mAP @ 0.50) TYLKO dla wybranych klas
    if valid_indices_for_mean:
        # Wybieramy z macierzy tylko kolumny odpowiadające klasom 1, 2, 3, 4
        filtered_precision_array = precision_array[:, valid_indices_for_mean]
        
        # Liczymy średnią po osi klas (axis=1) z przefiltrowanej tablicy
        mean_precision = np.mean(filtered_precision_array, axis=1)
        
        plt.plot(x_recall, mean_precision, label='mAP @ 0.50 (Mean 1-4)', 
                 linewidth=4, color='black', linestyle='--')
    else:
        print("Nie znaleziono żadnej z klas 1, 2, 3, 4 w wynikach ewaluacji.")

    # Kosmetyka wykresu
    plt.title('Krzywe Precision-Recall (Klasy 1, 2, 3, 4)', fontsize=14)
    plt.xlabel('Recall (Czułość)', fontsize=12)
    plt.ylabel('Precision (Precyzja)', fontsize=12)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Wykres zapisano jako: {output_image}")