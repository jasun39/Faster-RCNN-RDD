import unittest
import torch
import numpy as np
from model.utils import get_iou, boxes_to_transformation_targets, apply_regression_pred_to_anchors_or_proposals
from model.faster_rcnn import FasterRCNN

# Przykładowa konfiguracja (zmniejszona dla szybkości testów)
TEST_CONFIG = {
    'min_im_size': 600,
    'max_im_size': 1000,
    'backbone_out_channels': 2048,
    'scales': [128, 256, 512],
    'aspect_ratios': [0.5, 1, 2],
    'rpn_bg_threshold': 0.3,
    'rpn_fg_threshold': 0.7,
    'rpn_nms_threshold': 0.7,
    'rpn_train_prenms_topk': 100,
    'rpn_test_prenms_topk': 50,
    'rpn_train_topk': 50,
    'rpn_test_topk': 20,
    'rpn_batch_size': 256,
    'rpn_pos_fraction': 0.5,
    'roi_iou_threshold': 0.5,
    'roi_low_bg_iou': 0.0,
    'roi_pool_size': 7,
    'roi_nms_threshold': 0.3,
    'roi_topk_detections': 10,
    'roi_score_threshold': 0.05,
    'roi_batch_size': 32,
    'roi_pos_fraction': 0.25,
    'fc_inner_dim': 512
}

class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        # Ustawiamy ziarno dla powtarzalności
        torch.manual_seed(42)

    # TESTY MATEMATYCZNE
    
    def test_iou_calculation(self):
        """Sprawdza czy IoU jest liczone poprawnie."""
        # Box format: x1, y1, x2, y2
        box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        box2 = torch.tensor([[5, 0, 15, 10]], dtype=torch.float32) # Przesunięty o połowę w prawo
        
        # Intersection = 5x10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
        iou = get_iou(box1, box2)
        self.assertAlmostEqual(iou.item(), 1/3, places=4, msg="Błąd w obliczaniu IoU dla częściowego nakładania")

        # Test braku nakładania
        box3 = torch.tensor([[20, 20, 30, 30]], dtype=torch.float32)
        iou_zero = get_iou(box1, box3)
        self.assertEqual(iou_zero.item(), 0.0, msg="IoU powinno wynosić 0 dla rozłącznych ramek")

    def test_box_transformations(self):
        """Sprawdza czy kodowanie i dekodowanie ramek jest odwracalne (w przybliżeniu)."""
        anchors = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32) # 40x40
        gt_boxes = torch.tensor([[12, 12, 52, 52]], dtype=torch.float32) # Przesunięte
        
        deltas = boxes_to_transformation_targets(gt_boxes, anchors)
        
        deltas_expanded = deltas.unsqueeze(1) 
        reconstructed_boxes = apply_regression_pred_to_anchors_or_proposals(deltas_expanded, anchors)
        reconstructed_boxes = reconstructed_boxes.squeeze(1)
        
        # Czy odtworzona ramka jest bliska GT
        diff = torch.abs(reconstructed_boxes - gt_boxes).max().item()
        self.assertLess(diff, 1e-4, msg="Transformacja ramek nie jest odwracalna")

    # PRZEPŁYW DANYCH

    def test_forward_pass_training(self):
        """Sprawdza czy model w trybie treningowym zwraca słownik strat."""
        model = FasterRCNN(TEST_CONFIG, num_classes=2)
        model.train()
        
        # Dummy image (Batch=1, C=3, H=600, W=800)
        images = torch.rand(1, 3, 600, 800)
        
        # Dummy targets
        targets = {
            'bboxes': torch.tensor([[[10, 10, 50, 50], [100, 100, 200, 200]]], dtype=torch.float32), # 2 obiekty
            'labels': torch.tensor([[1, 1]], dtype=torch.int64) # Klasa 1
        }
        
        # Uruchomienie
        try:
            rpn_out, frcnn_out = model(images, targets)
        except RuntimeError as e:
            self.fail(f"Forward pass rzucił błędem: {e}")
            
        # Sprawdzenie czy mamy straty
        self.assertIn('rpn_classification_loss', rpn_out)
        self.assertIn('rpn_localization_loss', rpn_out)
        self.assertIn('frcnn_classification_loss', frcnn_out)
        self.assertIn('frcnn_localization_loss', frcnn_out)
        
        # Sprawdzenie czy straty są skalarami i mają gradient
        total_loss = rpn_out['rpn_classification_loss'] + frcnn_out['frcnn_localization_loss']
        self.assertTrue(total_loss.requires_grad, "Strata nie ma podpiętego gradientu (backprop nie zadziała!)")

    def test_forward_pass_inference(self):
        """Sprawdza czy model w trybie ewaluacji zwraca ramki."""
        model = FasterRCNN(TEST_CONFIG, num_classes=2)
        model.eval()
        
        images = torch.rand(1, 3, 600, 800)
        
        with torch.no_grad():
            rpn_out, frcnn_out = model(images)
            
        self.assertIn('boxes', frcnn_out)
        self.assertIn('scores', frcnn_out)
        self.assertIn('labels', frcnn_out)
        
        # Sprawdzenie kształtów
        num_dets = frcnn_out['boxes'].shape[0]
        self.assertEqual(frcnn_out['boxes'].shape, (num_dets, 4))
        self.assertEqual(frcnn_out['scores'].shape, (num_dets,))

    # OVERFIT TEST
    
    def test_overfitting_single_batch(self):
        """
        Próbuje nauczyć model jednego przykładu na pamięć. 
        Strata powinna drastycznie spaść.
        """
        print("\nUruchamiam test overfittingu (może potrwać kilka sekund)...")
        model = FasterRCNN(TEST_CONFIG, num_classes=2)
        model.train()
        
        # Zamrażamy backbone dla szybkości testu (uczymy tylko głowice)
        for p in model.backbone.parameters():
            p.requires_grad = False
            
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Stały obraz i target
        image = torch.rand(1, 3, 600, 600)
        targets = {
            'bboxes': torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32),
            'labels': torch.tensor([[1]], dtype=torch.int64)
        }
        
        initial_loss = float('inf')
        final_loss = 0.0
        
        # Krótka pętla treningowa (50 kroków powinno wystarczyć dla jednego przykładu)
        for i in range(50):
            optimizer.zero_grad()
            rpn_out, frcnn_out = model(image, targets)
            loss = rpn_out['rpn_classification_loss'] + rpn_out['rpn_localization_loss'] + \
                   frcnn_out['frcnn_classification_loss'] + frcnn_out['frcnn_localization_loss']
            
            if i == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
            
        print(f"Wejściowa wartość funkcji straty: {initial_loss:.4f}, końcowa wartość funkcji straty: {final_loss:.4f}")
        
        # Sprawdzamy czy strata spadła przynajmniej o 50% (dla jednego przykładu powinna spaść prawie do 0)
        self.assertLess(final_loss, initial_loss * 0.5, "Model nie potrafi overfitować jednego przykładu - poważny błąd w logice!")

if __name__ == '__main__':
    unittest.main()