"""
Approximate ODS / OIS edge-detection evaluator (pure Python, no MATLAB, no
compiled extensions).

This reimplements the core idea of the standard BSDS500 boundary benchmark
(as used by the PiDiNet/HED/RCF papers), but with one simplification: pixel
correspondence between predicted and ground-truth boundaries is computed via
nearest-neighbor distance (scipy.spatial.cKDTree) rather than the official
one-to-one bipartite graph matching (Piotr Dollar's correspondePixels, a
compiled MATLAB/C++ routine). Because of that, the numbers here are a close
APPROXIMATION of the official ODS/OIS, not a bit-identical reproduction.

Protocol:
  - For each test image, load the predicted probability map (.mat, key
    'img') and all human-annotator boundary maps (ground truth .mat).
  - Ground truth per image = pixel-wise UNION of all annotators' boundaries.
  - For a set of thresholds in (0, 1):
      - binarize the prediction at that threshold, thin to a 1px skeleton
      - precision = fraction of predicted-boundary pixels within maxDist of
        a GT pixel
      - recall    = fraction of GT pixels within maxDist of a predicted
        pixel
  - ODS: pool TP/FP/FN counts across the whole dataset at each threshold,
    take the threshold with the best dataset-level F-measure.
  - OIS: for each image independently pick its own best threshold, then
    average that image's best F-measure across the dataset.

Usage:
  python eval_ods_ois.py <mats_dir> <gt_mat_dir> [--max-dist-frac 0.0075] [--n-thresholds 33]
"""
import argparse
import glob
import os

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def load_prediction(mat_path):
    m = sio.loadmat(mat_path)
    return m['img'].astype(np.float32)


def load_gt_union(mat_path):
    m = sio.loadmat(mat_path)
    gt = m['groundTruth']
    n_annot = gt.shape[1]
    union = None
    for i in range(n_annot):
        b = gt[0, i][0, 0]['Boundaries'].astype(bool)
        union = b if union is None else (union | b)
    return union


def counts_at_threshold(pred_prob, gt_bin, threshold, max_dist_px, gt_tree, n_gt):
    pred_bin = pred_prob >= threshold
    if pred_bin.sum() == 0:
        return 0, 0, n_gt  # tp, fp(pred pixels)->0, fn = all gt pixels missed
    pred_thin = skeletonize(pred_bin)
    pred_pts = np.argwhere(pred_thin)
    n_pred = len(pred_pts)
    if n_pred == 0:
        return 0, 0, n_gt

    if n_gt == 0:
        return 0, n_pred, 0

    # precision side: for each predicted pixel, is there a GT pixel within max_dist?
    dist_to_gt, _ = gt_tree.query(pred_pts, k=1)
    tp_precision = int((dist_to_gt <= max_dist_px).sum())
    fp = n_pred - tp_precision

    # recall side: for each GT pixel, is there a predicted pixel within max_dist?
    pred_tree = cKDTree(pred_pts)
    gt_pts = gt_tree.data
    dist_to_pred, _ = pred_tree.query(gt_pts, k=1)
    tp_recall = int((dist_to_pred <= max_dist_px).sum())
    fn = n_gt - tp_recall

    # Use the recall-side TP for recall and precision-side TP for precision;
    # this matches how precision/recall are defined independently in the
    # official protocol (they need not be the exact same "TP" count).
    return (tp_precision, fp, fn, tp_recall)


def f_measure(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mats_dir', help='dir of predicted .mat files (key "img"), e.g. eval_results/mats_epoch_XXX')
    ap.add_argument('gt_mat_dir', help='dir of ground-truth .mat files (BSDS500 groundTruth/test)')
    ap.add_argument('--max-dist-frac', type=float, default=0.0075, help='tolerance as fraction of image diagonal (paper default 0.0075)')
    ap.add_argument('--n-thresholds', type=int, default=33)
    args = ap.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.mats_dir, '*.mat')))
    assert len(pred_files) > 0, 'no prediction .mat files found in %s' % args.mats_dir

    thresholds = np.linspace(0.02, 0.98, args.n_thresholds)

    # Per-threshold pooled counts across the dataset (for ODS)
    pooled_tp_p = np.zeros(len(thresholds))
    pooled_fp = np.zeros(len(thresholds))
    pooled_tp_r = np.zeros(len(thresholds))
    pooled_fn = np.zeros(len(thresholds))

    per_image_best_f = []
    n_images = 0

    for pf in pred_files:
        name = os.path.splitext(os.path.basename(pf))[0]
        gt_path = os.path.join(args.gt_mat_dir, name + '.mat')
        if not os.path.exists(gt_path):
            continue
        n_images += 1

        pred_prob = load_prediction(pf)
        gt_bin = load_gt_union(gt_path)
        if pred_prob.shape != gt_bin.shape:
            # some models output slightly different sizes; resize pred to GT
            from PIL import Image
            pred_prob = np.array(
                Image.fromarray(pred_prob).resize((gt_bin.shape[1], gt_bin.shape[0]), Image.BILINEAR)
            )

        H, W = gt_bin.shape
        diag = np.sqrt(H ** 2 + W ** 2)
        max_dist_px = args.max_dist_frac * diag

        gt_pts = np.argwhere(gt_bin)
        n_gt = len(gt_pts)
        gt_tree = cKDTree(gt_pts) if n_gt > 0 else None

        image_f_scores = []
        for ti, t in enumerate(thresholds):
            if n_gt == 0:
                continue
            res = counts_at_threshold(pred_prob, gt_bin, t, max_dist_px, gt_tree, n_gt)
            if len(res) == 3:
                tp_p, fp, fn = res
                tp_r = 0
            else:
                tp_p, fp, fn, tp_r = res

            n_pred_boundary = tp_p + fp
            precision = tp_p / n_pred_boundary if n_pred_boundary > 0 else 0.0
            recall = tp_r / n_gt if n_gt > 0 else 0.0
            image_f_scores.append(f_measure(precision, recall))

            pooled_tp_p[ti] += tp_p
            pooled_fp[ti] += fp
            pooled_tp_r[ti] += tp_r
            pooled_fn[ti] += fn

        if image_f_scores:
            per_image_best_f.append(max(image_f_scores))

        print('  [%d/%d] %s  n_gt=%d' % (n_images, len(pred_files), name, n_gt))

    # ODS: dataset-pooled precision/recall per threshold -> best F
    ods_f = []
    for ti in range(len(thresholds)):
        n_pred_boundary = pooled_tp_p[ti] + pooled_fp[ti]
        precision = pooled_tp_p[ti] / n_pred_boundary if n_pred_boundary > 0 else 0.0
        n_gt_total = pooled_tp_r[ti] + pooled_fn[ti]
        recall = pooled_tp_r[ti] / n_gt_total if n_gt_total > 0 else 0.0
        ods_f.append(f_measure(precision, recall))
    ods = max(ods_f)
    best_t_idx = int(np.argmax(ods_f))

    ois = float(np.mean(per_image_best_f)) if per_image_best_f else 0.0

    print('\n=== Results (approximate, nearest-neighbor matching) ===')
    print('N images evaluated : %d' % n_images)
    print('Best threshold      : %.3f' % thresholds[best_t_idx])
    print('ODS (dataset-level) : %.4f' % ods)
    print('OIS (per-image avg) : %.4f' % ois)


if __name__ == '__main__':
    main()
