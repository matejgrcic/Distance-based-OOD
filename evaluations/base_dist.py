import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
import torch.nn.functional as F

MAX_DIST = 50 # due to available disparity

class OODDistEvaluation:
    def __init__(self, loader, method, min=5, max=MAX_DIST, buckets=10):
        assert max <= MAX_DIST
        self.compute_ood_probs = method
        self.test_loader = loader
        self.ignore_id = loader.dataset.ignore_id
        self.buckets = []
        delta = int((max-min)/(buckets-1))
        start = min
        end = delta
        while end <= max:
            self.buckets.append((start, end))
            start = end
            end = end + delta

    def calculate_auroc(self,conf, gt):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        for i, j in zip(tpr, fpr):
            if i > 0.95:
                fpr_best = j
                break
        return roc_auc, fpr_best

    def calculate_ood_scores(self):
        total_conf = []
        total_gt = []
        total_dist = []
        for step, batch in enumerate(self.test_loader):
            img, lbl, dist = batch
            lbl = lbl[:, 0]
            lbl[lbl == 255] = self.ignore_id
            conf_probs = self.compute_ood_probs(img)

            label = lbl.view(-1)
            conf_probs = conf_probs.view(-1)
            dist = dist.view(-1)[label != self.ignore_id]
            total_dist.append(dist)
            gt = label[label != self.ignore_id].cpu()
            total_gt.append(gt)
            conf = conf_probs.cpu()[label != self.ignore_id]
            total_conf.append(conf)

        total_gt = torch.cat(total_gt, dim=0).numpy()
        total_conf = torch.cat(total_conf, dim=0).numpy()
        total_dist = torch.cat(total_dist, dim=0).numpy()
        output = []
        for min_b, max_b in self.buckets:
            idx = np.where(np.logical_and(total_dist >= min_b, total_dist < max_b))[0]
            if idx.shape[0] == 0:
                continue
            AP = average_precision_score(total_gt[idx], total_conf[idx])
            roc_auc, fpr = self.calculate_auroc(total_conf[idx], total_gt[idx])
            output.append((AP, fpr, roc_auc, (min_b, max_b)))
        return output



