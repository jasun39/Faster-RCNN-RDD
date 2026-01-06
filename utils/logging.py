import logging
import pandas as pd
import os

from torch.utils.tensorboard.writer import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def set_log(log_dir):
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # datefmt='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

def log(content, *args):
    for arg in args:
        content += str(arg)
    logger.info(content)

def csv_log(
    log_dir, 
    stats, 
    epoch,
    train_loss_list,
    loss_cls_list,
    loss_box_reg_list,
    loss_objectness_list,
    loss_rpn_list
):
    if epoch+1 == 1:
        create_log_csv(log_dir) 
    
    df = pd.DataFrame(
        {
            'epoch': int(epoch+1),
            'map_05': [float(stats[0])],
            'map': [float(stats[1])],
            'train loss': train_loss_list[-1],
            'train cls loss': loss_cls_list[-1],
            'train box reg loss': loss_box_reg_list[-1],
            'train obj loss': loss_objectness_list[-1],
            'train rpn loss': loss_rpn_list[-1]
        }
    )
    df.to_csv(
        os.path.join(log_dir, 'results.csv'), 
        mode='a', 
        index=False, 
        header=False
    )

def create_log_csv(log_dir):
    cols = [
        'epoch', 
        'map', 
        'map_05',
        'train loss',
        'train cls loss',
        'train box reg loss',
        'train obj loss',
        'train rpn loss'
    ]
    results_csv = pd.DataFrame(columns=cols)
    results_csv.to_csv(os.path.join(log_dir, 'results.csv'), index=False)

def coco_log(log_dir, stats):
    log_dict_keys = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    ]
    log_dict = {}
    # for i, key in enumerate(log_dict_keys):
    #     log_dict[key] = stats[i]

    with open(f"{log_dir}/train.log", 'a+') as f:
        f.writelines('\n')
        for i, key in enumerate(log_dict_keys):
            out_str = f"{key} = {stats[i]}"
            logger.debug(out_str) # DEBUG model so as not to print on console.
        logger.debug('\n'*2) # DEBUG model so as not to print on console.
    # f.close()

def tensorboard_loss_log(name, loss_np_arr, writer, epoch):
    """
    To plot graphs for TensorBoard log. The save directory for this
    is the same as the training result save directory.
    """
    writer.add_scalar(name, loss_np_arr[-1], epoch)

def tensorboard_map_log(name, val_map_05, val_map, writer, epoch):
    writer.add_scalars(
        name,
        {
            'mAP@0.5': val_map_05[-1], 
            'mAP@0.5_0.95': val_map[-1]
        },
        epoch
    )

def plot_confusion_matrix(cm, classes, normalize=False, title='Macierz Pomyłek', cmap=plt.cm.Blues):
    """
    Rysuje macierz pomyłek używając Matplotlib.
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    output_path = 'runs/confusion_matrix.png'
    plt.savefig(output_path)
    print(f"Wykres macierzy pomyłek zapisano w: {output_path}")
    plt.close()

def calculate_confusion_matrix_and_fps(model, data_loader, device, num_classes, iou_threshold=0.5, score_threshold=0.15):
    """
    Oblicza FPS i zbiera dane do macierzy pomyłek.
    """
    model.eval()
    
    # Macierz: wiersze=GT, kolumny=Pred. Ostatni indeks to 'Background' (nic nie wykryto / fałszywy alarm)
    # Rozmiar: (num_classes + 1) x (num_classes + 1)
    # Mapowanie: 0..N-1 to klasy obiektów, N to Tło.
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    
    total_time = 0.0
    total_images = 0

    print("\nObliczanie FPS i Macierzy Pomyłek...")
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(img.to(device) for img in images)
            
            # --- Pomiar czasu inferencji (FPS) ---
            start_time = time.time()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            total_images += len(images)
            
            # --- Logika Macierzy Pomyłek ---
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_labels = output['labels']

                gt_boxes = targets[i]['boxes'].to(device)
                gt_labels = targets[i]['labels'].to(device)

                # Filtrowanie po progu pewności (score threshold)
                mask = pred_scores >= score_threshold
                pred_boxes = pred_boxes[mask]
                pred_labels = pred_labels[mask]

                if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                    continue # True Negative (Tło vs Tło) - zazwyczaj ignorowane w detekcji

                if len(gt_boxes) == 0 and len(pred_boxes) > 0:
                    # Wszystkie predykcje to False Positives (Tło wykryte jako obiekt)
                    for label in pred_labels:
                        cm[num_classes, label.item() - 1] += 1 # GT=Background, Pred=Class
                    continue

                if len(gt_boxes) > 0 and len(pred_boxes) == 0:
                    # Wszystkie GT to False Negatives (Obiekt niewykryty)
                    for label in gt_labels:
                        cm[label.item() - 1, num_classes] += 1 # GT=Class, Pred=Background
                    continue

                # Obliczanie IoU między wszystkimi parami GT i Pred
                ious = box_iou(gt_boxes, pred_boxes)
                
                # Dopasowanie ramek
                matched_gt = set()
                matched_pred = set()
                
                # Przechodzimy po IoU od najwyższego
                if ious.numel() > 0:
                    vals, indices = torch.sort(ious.flatten(), descending=True)
                    
                    for val, idx in zip(vals, indices):
                        if val < iou_threshold:
                            break
                        
                        gt_idx = (idx // ious.shape[1]).item()
                        pred_idx = (idx % ious.shape[1]).item()
                        
                        if gt_idx in matched_gt or pred_idx in matched_pred:
                            continue
                        
                        # Mamy dopasowanie geometryczne
                        matched_gt.add(gt_idx)
                        matched_pred.add(pred_idx)
                        
                        gt_cls = gt_labels[gt_idx].item() - 1 # -1 bo klasy są 1-indexed, a macierz 0-indexed
                        pred_cls = pred_labels[pred_idx].item() - 1
                        
                        cm[gt_cls, pred_cls] += 1

                # Obsługa nieprzypisanych (FN i FP)
                for i in range(len(gt_boxes)):
                    if i not in matched_gt:
                        gt_cls = gt_labels[i].item() - 1
                        cm[gt_cls, num_classes] += 1 # FN: GT=Class, Pred=Background

                for i in range(len(pred_boxes)):
                    if i not in matched_pred:
                        pred_cls = pred_labels[i].item() - 1
                        cm[num_classes, pred_cls] += 1 # FP: GT=Background, Pred=Class

    avg_fps = total_images / total_time
    return avg_fps, cm