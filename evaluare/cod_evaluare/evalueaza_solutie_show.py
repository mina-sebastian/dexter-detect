import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    return inter_area / float(box_a_area + box_b_area - inter_area)

def compute_average_precision(rec, prec):
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))

    # Loop only up to the second-to-last index to avoid going out of bounds
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])

    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    return np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])


def eval_detections(detections, scores, file_names, ground_truth_path,
                    show_images=False, images_folder="images"):
    """ Evaluate detections with precision, recall, AP,
        and optionally display each image with boxes:
         - Gray for ground-truth
         - Green for true positives
         - Red for false positives
    """
    # Load ground truth
    gt_file = np.loadtxt(ground_truth_path, dtype=str)
    gt_file_names = gt_file[:, 0]
    gt_boxes = np.array(gt_file[:, 1:], np.int64)

    num_gt = len(gt_boxes)
    gt_used = np.zeros(num_gt)

    # Sort detections by score descending
    sorted_indices = np.argsort(scores)[::-1]
    detections = detections[sorted_indices]
    scores = scores[sorted_indices]
    file_names = file_names[sorted_indices]

    num_det = len(detections)
    tp = np.zeros(num_det)
    fp = np.zeros(num_det)

    # For visualization: store GT, TPs, and FPs keyed by filename
    # We'll gather them and plot after the loop if show_images=True
    from collections import defaultdict
    vis_data = defaultdict(lambda: {"gt": [], "tp": [], "fp": []})

    # Mark ground-truth for each image
    for i, fname in enumerate(gt_file_names):
        vis_data[fname]["gt"].append(gt_boxes[i])

    # Go through detections
    for i in range(num_det):
        fname = file_names[i]
        box = detections[i]

        # find all GT for this image
        idxs = np.where(gt_file_names == fname)[0]
        gt_for_image = gt_boxes[idxs]

        max_overlap = -1
        max_idx = -1
        for gt_i, gt_box in enumerate(gt_for_image):
            iou = intersection_over_union(box, gt_box)
            if iou > max_overlap:
                max_overlap = iou
                max_idx = idxs[gt_i]

        if max_overlap >= 0.3:
            # Not used yet? => true positive; else => duplicated => false positive
            if gt_used[max_idx] == 0:
                tp[i] = 1
                gt_used[max_idx] = 1
                vis_data[fname]["tp"].append(box)
            else:
                fp[i] = 1
                vis_data[fname]["fp"].append(box)
        else:
            fp[i] = 1
            vis_data[fname]["fp"].append(box)

    # Compute precision / recall / AP
    cum_fp = np.cumsum(fp)
    cum_tp = np.cumsum(tp)
    rec = cum_tp / num_gt
    prec = cum_tp / (cum_tp + cum_fp)
    ap = compute_average_precision(rec, prec)

    # Plot the PR curve
    plt.figure()
    plt.plot(rec, prec, '-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'All faces: AP={ap:.3f}')
    plt.show()

    # If asked, display each image with bounding boxes
    if show_images:
        for fname in vis_data:
            img_path = os.path.join(images_folder, fname)
            if not os.path.exists(img_path):
                # If you don't have actual images, just skip
                continue

            img = mpimg.imread(img_path)
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw GT in gray
            for g in vis_data[fname]["gt"]:
                x1, y1, x2, y2 = g
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                         linewidth=2, edgecolor='gray', fill=False)
                ax.add_patch(rect)

            # Draw TP in green
            for t in vis_data[fname]["tp"]:
                x1, y1, x2, y2 = t
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                         linewidth=2, edgecolor='green', fill=False)
                ax.add_patch(rect)

            # Draw FP in red
            for f in vis_data[fname]["fp"]:
                x1, y1, x2, y2 = f
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                         linewidth=2, edgecolor='red', fill=False)
                ax.add_patch(rect)

            plt.title(fname)
            plt.show()

def eval_detections_character(detections, scores, file_names, ground_truth_path,
                             character, show_images=False, images_folder="images"):
    """ Same logic, adapted for a specific character. """
    # (For brevity, this is basically identical to eval_detections,
    #  except for the final plot title. Reuse the same approach.)
    gt_file = np.loadtxt(ground_truth_path, dtype=str)
    gt_file_names = gt_file[:, 0]
    gt_boxes = np.array(gt_file[:, 1:], np.int64)

    num_gt = len(gt_boxes)
    gt_used = np.zeros(num_gt)

    sorted_indices = np.argsort(scores)[::-1]
    detections = detections[sorted_indices]
    scores = scores[sorted_indices]
    file_names = file_names[sorted_indices]

    num_det = len(detections)
    tp = np.zeros(num_det)
    fp = np.zeros(num_det)

    from collections import defaultdict
    vis_data = defaultdict(lambda: {"gt": [], "tp": [], "fp": []})

    for i, fname in enumerate(gt_file_names):
        vis_data[fname]["gt"].append(gt_boxes[i])

    for i in range(num_det):
        fname = file_names[i]
        box = detections[i]
        idxs = np.where(gt_file_names == fname)[0]
        gt_for_image = gt_boxes[idxs]

        max_overlap = -1
        max_idx = -1
        for gt_i, gt_box in enumerate(gt_for_image):
            iou = intersection_over_union(box, gt_box)
            if iou > max_overlap:
                max_overlap = iou
                max_idx = idxs[gt_i]

        if max_overlap >= 0.3:
            if gt_used[max_idx] == 0:
                tp[i] = 1
                gt_used[max_idx] = 1
                vis_data[fname]["tp"].append(box)
            else:
                fp[i] = 1
                vis_data[fname]["fp"].append(box)
        else:
            fp[i] = 1
            vis_data[fname]["fp"].append(box)

    cum_fp = np.cumsum(fp)
    cum_tp = np.cumsum(tp)
    rec = cum_tp / num_gt
    prec = cum_tp / (cum_tp + cum_fp)
    ap = compute_average_precision(rec, prec)

    plt.figure()
    plt.plot(rec, prec, '-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{character} faces: AP={ap:.3f}')
    plt.show()

    if show_images:
        for fname in vis_data:
            img_path = os.path.join(images_folder, fname)
            if not os.path.exists(img_path):
                continue

            img = mpimg.imread(img_path)
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            for g in vis_data[fname]["gt"]:
                x1, y1, x2, y2 = g
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                         linewidth=2, edgecolor='gray', fill=False)
                ax.add_patch(rect)
            for t in vis_data[fname]["tp"]:
                x1, y1, x2, y2 = t
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                         linewidth=2, edgecolor='green', fill=False)
                ax.add_patch(rect)
            for f in vis_data[fname]["fp"]:
                x1, y1, x2, y2 = f
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                         linewidth=2, edgecolor='red', fill=False)
                ax.add_patch(rect)

            plt.title(f"{character} - {fname}")
            plt.show()

def evaluate_results_task1(solution_path, ground_truth_path, verbose=0):
    detections = np.load(solution_path + "detections_all_faces.npy",
                         allow_pickle=True, fix_imports=True, encoding='latin1')
    scores = np.load(solution_path + "scores_all_faces.npy",
                     allow_pickle=True, fix_imports=True, encoding='latin1')
    file_names = np.load(solution_path + "file_names_all_faces.npy",
                         allow_pickle=True, fix_imports=True, encoding='latin1')

    # Pass show_images=True and an images_folder if you want to see the bounding boxes
    eval_detections(detections, scores, file_names, ground_truth_path,
                    show_images=True, images_folder="testare")

def evaluate_results_task2(solution_path, ground_truth_path, character, verbose=0):
    detections = np.load(solution_path + f"detections_{character}.npy",
                         allow_pickle=True, fix_imports=True, encoding='latin1')
    scores = np.load(solution_path + f"scores_{character}.npy",
                     allow_pickle=True, fix_imports=True, encoding='latin1')
    file_names = np.load(solution_path + f"file_names_{character}.npy",
                         allow_pickle=True, fix_imports=True, encoding='latin1')

    eval_detections_character(detections, scores, file_names,
                              ground_truth_path, character,
                              show_images=True, images_folder="testare")

verbose = 1
solution_path_root = "data\\output\\"
ground_truth_path_root = "validare\\"

# Task1
solution_path = solution_path_root + "task1_temp\\"
ground_truth_path = ground_truth_path_root + "task1_gt_validare.txt"
evaluate_results_task1(solution_path, ground_truth_path, verbose)

# Task2
solution_path = solution_path_root + "task2_temp\\"

for ch in ["dad", "mom", "dexter", "deedee"]:
    gt_path = ground_truth_path_root + f"task2_{ch}_gt_validare.txt"
    evaluate_results_task2(solution_path, gt_path, ch, verbose)
