import torch
import torch.nn as nn
import torchvision
from .utils import (get_iou, boxes_to_transformation_targets, 
                   apply_regression_pred_to_anchors_or_proposals, 
                   sample_positive_negative, clamp_boxes_to_image_boundary)

class ROIHead(nn.Module):
    """
    Głowa ROI nad warstwą ROI pooling do generowania
    przewidywań klasyfikacji i transformacji ramek
    """
    
    def __init__(self, model_config, num_classes, in_channels):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.roi_batch_size = model_config['roi_batch_size']
        self.roi_pos_count = int(model_config['roi_pos_fraction'] * self.roi_batch_size)
        self.iou_threshold = model_config['roi_iou_threshold']
        self.low_bg_iou = model_config['roi_low_bg_iou']
        self.nms_threshold = model_config['roi_nms_threshold']
        self.topK_detections = model_config['roi_topk_detections']
        self.low_score_threshold = model_config['roi_score_threshold']
        self.pool_size = model_config['roi_pool_size']
        self.fc_inner_dim = model_config['fc_inner_dim']
        

        # Warstwy klasyfikatorów klas i obszarów
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)
        
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)

        torch.nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_reg_layer.bias, 0)
    
    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        Przypisz propozycje do ramek gt lub tła na podstawie IOU
        """
        if gt_boxes.numel() == 0:
            # Sytuacja: Brak obiektów na zdjęciu (samo tło)
            device = proposals.device
            # Wszystkie etykiety ustawiamy na 0 (tło)
            labels = torch.zeros(proposals.shape[0], dtype=torch.int64, device=device)
            # Ramki dopasowane to same zera (nie będą używane do straty lokalizacji, bo etykieta to 0)
            matched_gt_boxes = torch.zeros_like(proposals)
            return labels, matched_gt_boxes
        
        iou_matrix = get_iou(gt_boxes, proposals)
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        background_proposals = (best_match_iou < self.iou_threshold) & (best_match_iou >= self.low_bg_iou)
        ignored_proposals = best_match_iou < self.low_bg_iou
        
        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2
        
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        labels[background_proposals] = 0
        labels[ignored_proposals] = -1
        
        return labels, matched_gt_boxes_for_proposals
    
    def forward(self, feat, proposals, image_shapes, targets=None):
        """
        Args:
            feat: Tensor [Batch, C, H, W]
            proposals: List[Tensor] - lista propozycji dla każdego obrazu
            image_shapes: List[Tuple] - lista wymiarów obrazów
            targets: List[Dict] - lista targetów (opcjonalnie)
        """
        final_proposals = []
        final_labels = []
        final_regression_targets = []
        
        if self.training and targets is not None:
            for i in range(len(proposals)):
                proposals_i = proposals[i]
                gt_boxes = targets[i]['bboxes']
                gt_labels = targets[i]['labels']
                
                proposals_i = torch.cat([proposals_i, gt_boxes], dim=0)
                
                labels, matched_gt_boxes = self.assign_target_to_proposals(proposals_i, gt_boxes, gt_labels)
                
                sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                    labels,
                    positive_count=self.roi_pos_count,
                    total_count=self.roi_batch_size
                )
                
                sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
                
                proposals_i = proposals_i[sampled_idxs]
                labels = labels[sampled_idxs]
                matched_gt_boxes = matched_gt_boxes[sampled_idxs]
                
                regression_targets_i = boxes_to_transformation_targets(matched_gt_boxes, proposals_i)
                
                final_proposals.append(proposals_i)
                final_labels.append(labels)
                final_regression_targets.append(regression_targets_i)
        else:
            # W trybie testowym po prostu bierzemy propozycje z RPN
            final_proposals = proposals

        scale_h = feat.shape[-2] / float(feat.shape[-2] * 16)
        spatial_scale = 1.0 / 32.0 
        
        proposal_roi_pool_feats = torchvision.ops.roi_pool(
            feat, 
            final_proposals, 
            output_size=self.pool_size,
            spatial_scale=spatial_scale
        )
        
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
        
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)
        
        frcnn_output = {}
        
        # Obliczanie Straty (Training)
        if self.training and targets is not None:
            all_labels = torch.cat(final_labels, dim=0)
            all_regression_targets = torch.cat(final_regression_targets, dim=0)
            
            classification_loss = torch.nn.functional.cross_entropy(cls_scores, all_labels)
            
            fg_proposals_idxs = torch.where(all_labels > 0)[0]
            fg_cls_labels = all_labels[fg_proposals_idxs]
            
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_pred[fg_proposals_idxs, fg_cls_labels],
                all_regression_targets[fg_proposals_idxs],
                beta=1/9,
                reduction="sum",
            )
            localization_loss = localization_loss / (all_labels.numel() + 1e-5)
            
            frcnn_output['frcnn_classification_loss'] = classification_loss
            frcnn_output['frcnn_localization_loss'] = localization_loss
            
            return frcnn_output
        
        else:
            boxes_per_image = [len(p) for p in final_proposals]
            
            pred_boxes_list = box_transform_pred.split(boxes_per_image, 0)
            pred_scores_list = cls_scores.split(boxes_per_image, 0)
            
            all_final_boxes = []
            all_final_scores = []
            all_final_labels = []
            
            for i, (pred_boxes_i, pred_scores_i) in enumerate(zip(pred_boxes_list, pred_scores_list)):
                # Logika dla pojedynczego zdjęcia
                proposals_i = final_proposals[i]
                image_shape_i = image_shapes[i]
                
                # Aplikacja regresji
                pred_boxes_i = apply_regression_pred_to_anchors_or_proposals(pred_boxes_i, proposals_i)
                pred_scores_i = torch.nn.functional.softmax(pred_scores_i, dim=-1)
                
                # Przycinanie do image_shape
                pred_boxes_i = clamp_boxes_to_image_boundary(pred_boxes_i, image_shape_i)
                
                # Rozpakowanie klas, pomijamy klasę tła (0)
                pred_labels_i = torch.arange(num_classes, device=feat.device)
                pred_labels_i = pred_labels_i.view(1, -1).expand_as(pred_scores_i)
                
                pred_boxes_i = pred_boxes_i[:, 1:].reshape(-1, 4)
                pred_scores_i = pred_scores_i[:, 1:].reshape(-1)
                pred_labels_i = pred_labels_i[:, 1:].reshape(-1)
                
                # Filtrowanie
                final_boxes, final_labels, final_scores = self.filter_predictions(
                    pred_boxes_i, pred_labels_i, pred_scores_i
                )
                
                all_final_boxes.append(final_boxes)
                all_final_scores.append(final_scores)
                all_final_labels.append(final_labels)
            
            frcnn_output['boxes'] = all_final_boxes 
            frcnn_output['scores'] = all_final_scores
            frcnn_output['labels'] = all_final_labels
            
            return frcnn_output

    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        # Usuń ramki z niskim wynikiem
        keep = torch.where(pred_scores > self.low_score_threshold)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # Usuń małe ramki
        min_size = 16
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # NMS osobno dla każdej klasy
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(pred_boxes[curr_indices],
                                                          pred_scores[curr_indices],
                                                          self.nms_threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
        keep = post_nms_keep_indices[:self.topK_detections]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        return pred_boxes, pred_labels, pred_scores