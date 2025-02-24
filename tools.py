import os
import cv2 as cv
import numpy as np
from PIL import ImageOps, Image
from numpy.lib.stride_tricks import as_strided

train_folder = './antrenare'
annots_files = [
        'dad_annotations.txt',
        'deedee_annotations.txt',
        'dexter_annotations.txt',
        'mom_annotations.txt',
    ]

img_dict = {}
means_dict = {}
annots_dict = {}

personages_mean_face_ratio = {
    'dad': 0.87,
    'deedee': 1.48,
    'dexter': 1.3,
    'mom': 0.960,
    'unknown': 1.094
}

import os
import cv2
import random


def export_to_yolo(ann_parser, output_dir="yolo_dataset"):
    label_map = {
        'mom': 0,
        'dad': 1,
        'dexter': 2,
        'deedee': 3,
        'unknown': 4
    }

    file_to_anns = {}
    for lbl, ann_list in ann_parser.annotations.items():
        for ann in ann_list:
            img_path = ann.get_file()
            file_to_anns.setdefault(img_path, []).append(ann)

    all_files = list(file_to_anns.keys())
    random.shuffle(all_files)

    train_count = int(len(all_files) * 0.8)
    train_files = all_files[:train_count]
    val_files = all_files[train_count:]

    img_train_dir = os.path.join(output_dir, 'images', 'train')
    img_val_dir   = os.path.join(output_dir, 'images', 'val')
    lbl_train_dir = os.path.join(output_dir, 'labels', 'train')
    lbl_val_dir   = os.path.join(output_dir, 'labels', 'val')
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    def write_to_yolo(files_list, images_dir, labels_dir):
        for img_path in files_list:
            img = cv2.imread(img_path)
            if img is None:
                continue

            if 'dad' in img_path:
                label = 'dad'
            elif 'deedee' in img_path:
                label = 'deedee'
            elif 'dexter' in img_path:
                label = 'dexter'
            elif 'mom' in img_path:
                label = 'mom'
            else:
                label = 'unknown'
            h, w = img.shape[:2]
            base_name       = label + '_' + os.path.basename(img_path)
            base_name_noext = os.path.splitext(base_name)[0]
            out_img_path    = os.path.join(images_dir, base_name)

            cv2.imwrite(out_img_path, img)

            label_file_path = os.path.join(labels_dir, base_name_noext + '.txt')
            with open(label_file_path, 'w') as f:
                for ann in file_to_anns[img_path]:
                    x1, y1, x2, y2 = ann.bbox
                    center_x = ((x1 + x2) / 2) / w
                    center_y = ((y1 + y2) / 2) / h
                    box_w    = (x2 - x1) / w
                    box_h    = (y2 - y1) / h

                    class_id = label_map.get(ann.label, label_map['unknown'])

                    f.write(f"{class_id} {center_x} {center_y} {box_w} {box_h}\n")

    write_to_yolo(train_files, img_train_dir, lbl_train_dir)
    write_to_yolo(val_files,   img_val_dir,   lbl_val_dir)

    print(f"{len(train_files)} train, {len(val_files)} val.")

def add_to_dict(key, value, img_dict=img_dict):
    if key not in img_dict:
        img_dict[key] = []
    img_dict[key].append(value)

def start():
    for annot_file in annots_files:
        with open(os.path.join(train_folder, annot_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(' ')
                image = cv.imread(os.path.join(train_folder, annot_file.split("_")[0], line[0]))
                x1 = int(line[1])
                y1 = int(line[2])
                x2 = int(line[3])
                y2 = int(line[4])
                fin = {
                    'name': line[5] + '_' + line[0],
                    'img': image[y1:y2, x1:x2],
                    'original': image,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                }
                add_to_dict(line[5], fin)
                add_to_dict(os.path.join(train_folder, annot_file.split("_")[0], line[0]), [x1, y1, x2, y2], annots_dict)

            

def calc_mean_dimensions():
    for key in img_dict:
        sum_x = 0
        sum_y = 0
        for img in img_dict[key]:
            sum_x += img['x2'] - img['x1']
            sum_y += img['y2'] - img['y1']
        means_dict[key] = (sum_x // len(img_dict[key]), sum_y // len(img_dict[key]))
        print(key, means_dict[key])

allMeansMean = []

def resize_to_mean():
    folder_to_save = './data/exemplePozitive'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    global allMeansMean
    allMeansMean = [
        sum([means_dict[key][0] for key in means_dict]) // len(means_dict),
        sum([means_dict[key][1] for key in means_dict]) // len(means_dict)
    ]

    print(allMeansMean)

    for key in img_dict:
        for img in img_dict[key]:
            cv.imwrite(os.path.join(folder_to_save, img['name']), cv.resize(img['img'], means_dict[key]))
            cv.imwrite(os.path.join(folder_to_save, 'all_' + img['name']), cv.resize(img['img'], (224, 224)))



def sliding_window(image, step_size_w, step_size_h, window_sizes):
    for window_size in window_sizes:
        for y in range(0, image.shape[0] - window_size[1] + 1, step_size_h):
            for x in range(0, image.shape[1] - window_size[0] + 1, step_size_w):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]], window_size)

def save_positive_examples():
    folder_to_save = './data/exemplePozitive'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    for key in img_dict:
        for img in img_dict[key]:
            cv.imwrite(os.path.join(folder_to_save, img['name']), img['img'])
            cv.imwrite(os.path.join(folder_to_save, 'all_' + img['name']), cv.resize(img['img'], (224, 224)))

# check what percent of the area of the first bbox is overlapped by the second bbox
def iou_simple(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return max(intersection / area1, intersection / area2)

def save_negative_examples(max_iou=0):
    folder_to_save = './data/exempleNegative'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)


    nb = 0
    for key in annots_dict:
        image = cv.imread(key)
        faces_annotations = annots_dict[key]
        windows = []
        winsz = 66
        for (x, y, window) in sliding_window(image, step_size=32, window_size=(winsz, winsz)):
            ious = [intersection_over_union([x, y, x + winsz, y + winsz], face) for face in faces_annotations]
            biggest_iou = max(ious) if ious else 0
            if biggest_iou <= max_iou:
                windows.append([x, y, x + winsz, y + winsz, biggest_iou])
        # save top k windows, sorted by the highest iou
        k = 2
        windows_sorted = sorted(windows, key=lambda x: x[4], reverse=False)
        np.random.shuffle(windows_sorted)
        windows = windows_sorted[:k]

        for i, (x1, y1, x2, y2, iou) in enumerate(windows):
            print('saving', iou)
            new_img = image[y1:y2, x1:x2]
            new_img = cv.resize(new_img, (224, 224))
            cv.imwrite(os.path.join(folder_to_save, 'i_' + key.split('/')[-1] + str(nb) + '_' + str(i) + '.jpg'), new_img)
        
        nb += 1
            

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

def translateMap(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    valueScaled = float(value - leftMin) / float(leftSpan)

    rez = rightMin + (valueScaled * rightSpan)
    print('rez:', rez)
    return rez


def non_maximal_suppression(image_detections, image_scores, image_size, iou_threshold=0.3):
    """
    Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
    Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
    fi in interiorul celeilalte detectii.
    :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
    :param image_scores: numpy array de dimensiune N
    :param image_size: tuplu, dimensiunea imaginii
    :return: image_detections si image_scores care sunt maximale.
    """
    # xmin, ymin, xmax, ymax
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    print(x_out_of_bounds, y_out_of_bounds)
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]
    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                    if intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                    else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                            is_maximal[j] = False
    return sorted_image_detections[is_maximal], sorted_scores[is_maximal]


def generate_negative_examples():
    max_overlap = 0
    os.makedirs('./data/exempleNegative', exist_ok=True)
    for key in annots_dict:
        image = cv.imread(key)
        for i in range(6):
            for mean in means_dict:
                x1 = np.random.randint(0, image.shape[1] - means_dict[mean][0])
                y1 = np.random.randint(0, image.shape[0] - means_dict[mean][1])
                x2 = x1 + means_dict[mean][0]
                y2 = y1 + means_dict[mean][1]
                bbox = [x1, y1, x2, y2]
                overlap = False
                for annot in annots_dict[key]:
                    if intersection_over_union(bbox, annot) > max_overlap:
                        overlap = True
                        break
                if not overlap:
                    cv.imwrite(os.path.join('./data/exempleNegative', key.split('/')[-1] + '_' + str(i) + '.jpg'), image[y1:y2, x1:x2])
            x1 = np.random.randint(0, image.shape[1] - allMeansMean[0])
            y1 = np.random.randint(0, image.shape[0] - allMeansMean[1])
            x2 = x1 + allMeansMean[0]
            y2 = y1 + allMeansMean[1]
            bbox = [x1, y1, x2, y2]
            overlap = False
            for annot in annots_dict[key]:
                if intersection_over_union(bbox, annot) > max_overlap:
                    overlap = True
                    break
            if not overlap:
                cv.imwrite(os.path.join('./data/exempleNegative', 'all_'+key.split('/')[-1] + '_' + str(i) + '.jpg'), image[y1:y2, x1:x2])
            
            

def print_face_ratios():
    means_dict = {}
    for key in img_dict:
        for img in img_dict[key]:
            print((img['x2'] - img['x1']) / (img['y2'] - img['y1']))
            if key not in means_dict:
                means_dict[key] = []
            means_dict[key].append((img['x2'] - img['x1']) / (img['y2'] - img['y1']))
    for key in means_dict:
        print(key, sum(means_dict[key]) / len(means_dict[key]))

def calculate_entropy(image):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-5))
    return entropy

def calculate_edge_density(image):
    edges = cv.Canny(image, 100, 200)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])
    return edge_density

def process_images(folder_path, threshold_entropy=5.0, threshold_edge_density=0.3):
    complex_images = []
    low_complexity_images = []
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            path = os.path.join(folder_path, file)
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            entropy = calculate_entropy(image)
            edge_density = calculate_edge_density(image)

            if entropy > threshold_entropy or edge_density > threshold_edge_density:
                complex_images.append(file)
            else:
                low_complexity_images.append(file)

    return complex_images, low_complexity_images

def show_high_entropy_images():
    folder_path = './data/exempleNegativeInit'
    folder_path2 = './data/exempleNegative'
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    high_complexity_images, low_c_imgs = process_images(folder_path)
    print('High complexity images:', len(high_complexity_images))
    print('Low complexity images:', len(low_c_imgs))
    for i, image in enumerate(high_complexity_images):
        img = cv.imread(os.path.join(folder_path, image))
        cv.imwrite(os.path.join('./data/exempleNegative', str(i)+'.jpg'), img)

def includes(n, n1, n2):
    included = (n1+1) < n and n < (n2-1)
    if included:
        print(n, n1, n2)
    return included

def point_in_box(x, y, box):
    return includes(x, box[0], box[2]) and includes(y, box[1], box[3])

windows = [(8 * 16, 8 * 16), (8 * 10, 8 * 16), (8 * 16, 8 * 10)]

def gen_negative_hard_examples():
    folder_path = './data/exempleNegativeHARDv3'
    for key in annots_dict:
        image = cv.imread(key)
        print('Processing', key)
        all_faces = annots_dict[key]
        
        found = []
        for face in all_faces:
            for window in windows:
                min_x = face[0]
                min_y = face[1]
                max_x = face[2]
                max_y = face[3]

                box_to_search_dims = (window[0], window[1])
                print('box_to_search_dims', box_to_search_dims)
                #up
                for i in range(0, max_x, box_to_search_dims[0]):
                    chX1 = min_x + i
                    chX2 = min_x + box_to_search_dims[0] + i
                    chY1 = min_y - box_to_search_dims[1]
                    chY2 = min_y
                    if chY1 < 0 or chX2 > image.shape[1]:
                        break
                    else:
                        found.append([chX1, chY1, chX2, chY2])
                #down
                for i in range(0, max_x, box_to_search_dims[0]):
                    chX1 = min_x + i
                    chX2 = min_x + box_to_search_dims[0] + i
                    chY1 = max_y
                    chY2 = max_y + box_to_search_dims[1]
                    if chY2 > image.shape[0] or chX2 > image.shape[1]:
                        break
                    else:
                        found.append([chX1, chY1, chX2, chY2])

                #left
                for i in range(0, max_y, box_to_search_dims[1]):
                    chY1 = min_y + i
                    chY2 = min_y + box_to_search_dims[1] + i
                    chX1 = min_x - box_to_search_dims[0]
                    chX2 = min_x
                    if chX1 < 0 or chY2 > image.shape[0]:
                        break
                    else:
                        found.append([chX1, chY1, chX2, chY2])
                #right
                for i in range(0, max_y, box_to_search_dims[1]):
                    chY1 = min_y + i
                    chY2 = min_y + box_to_search_dims[1] + i
                    chX1 = max_x
                    chX2 = max_x + box_to_search_dims[0]
                    if chX2 > image.shape[1] or chY2 > image.shape[0]:
                        break
                    else:
                        found.append([chX1, chY1, chX2, chY2])
            
        #clean found not to overlap with an adnotated face
        to_remove = []
        for i, face in enumerate(found):
            for already in all_faces:
                x = face[0]
                y = face[1]
                x2 = face[2]
                y2 = face[3]

                a_x = already[0]
                a_y = already[1]
                a_x2 = already[2]
                a_y2 = already[3]
                
                if point_in_box(x, y, already) or point_in_box(x2, y, already) or point_in_box(x, y2, already) or point_in_box(x2, y2, already):
                    to_remove.append(i)
                    break
                if point_in_box(a_x, a_y, face) or point_in_box(a_x2, a_y, face) or point_in_box(a_x, a_y2, face) or point_in_box(a_x2, a_y2, face):
                    to_remove.append(i)
                    break
        found = [found[i] for i in range(len(found)) if i not in to_remove]

        to_remove = []
        for i, face1 in enumerate(found):
            for j, face2 in enumerate(found):
                if i == j or i in to_remove or j in to_remove:
                    continue
                new_min_x = face1[0] + 2
                new_min_y = face1[1] + 2
                new_min_y2 = face1[3] - 2
                new_min_x2 = face1[2] - 2
                if intersection_over_union([new_min_x, new_min_y, new_min_x2, new_min_y2], face2) < 0.001:
                    continue
                if calculate_entropy(image[face1[1]:face1[3], face1[0]:face1[2]]) < calculate_entropy(image[face2[1]:face2[3], face2[0]:face2[2]]):
                    to_remove.append(i)
                    print('removing', i)
                    break
                else:
                    to_remove.append(j)
                    print('removing', j)
            if i in to_remove:
                break

        found = [found[i] for i in range(len(found)) if i not in to_remove]
        for i, face in enumerate(found):
            #show the boxes
            # cv.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)
            pth = 'dad'
            if 'mom' in key:
                pth = 'mom'
            elif 'deedee' in key:
                pth = 'deedee'
            elif 'dexter' in key:
                pth = 'dexter'
            pth_to_save = os.path.join(folder_path, pth)
            if not os.path.exists(pth_to_save):
                os.makedirs(pth_to_save)
            cv.imwrite(os.path.join(folder_path, pth, key.split('/')[-1] + '_' + str(i) + '.jpg'), image[face[1]:face[3], face[0]:face[2]])
            

        # for already in all_faces:
        #     cv.rectangle(image, (already[0], already[1]), (already[2], already[3]), (255, 0, 0), 2)

        
        # cv.imshow('image', image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()



def get_top_negative():
    folder_path = './data/exempleNegative'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path2 = './data/exempleNegativeHARDv3'
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    for key in ['mom', 'dad', 'deedee', 'dexter']:
        list_of_files = {}
        for file in os.listdir(os.path.join(folder_path2, key)):
            image = cv.imread(os.path.join(folder_path2, key, file))
            list_of_files[file] = {
                'entropy': calculate_entropy(image),
                'edge_density': calculate_edge_density(image),
                'image': image
            }
        sorted_files = sorted(list_of_files.items(), key=lambda x: x[1]['entropy'], reverse=True)
        # sorted_files = [x for x in sorted_files if x[1]['image'].shape[0] >= 66 ]
        for i, (file, _) in enumerate(sorted_files):
            # img = cv.resize(list_of_files[file]['image'], (66, 66))
            cv.imwrite(os.path.join(folder_path, 'top_' + key + '_' + str(i) + '.jpg'), list_of_files[file]['image'])

def to_str(w_shape):
    return str(w_shape[0]) + '_' + str(w_shape[1])

def hard_negatives_shapes():
    hard_path = './data/exempleNegativeHARDv3'
    key = {}
    for folder in os.listdir(hard_path):
        for file in os.listdir(os.path.join(hard_path, folder)):
            img = cv.imread(os.path.join(hard_path, folder, file))
            # print(img.shape)
            s = to_str(img.shape)
            if key.get(s) is None:
                key[s] = 0
            key[s] += 1
    print(key)

def combine_hard_examples(path1, path2, pref):
    for file in os.listdir(path1):
        img = cv.imread(os.path.join(path1, file))
        cv.imwrite(os.path.join(path2, pref + file), img)

if __name__ == '__main__':
    # hard_negatives_shapes()
    # combine_hard_examples('data\\hard_examples', 'data\\hard_examples_old1', 'deedee_')
    # start()
    # get_top_negative()
    # gen_negative_hard_examples()
    # print_face_ratios()
    # show_high_entropy_images()
    # save_positive_examples()
    # save_negative_examples()
    # calc_mean_dimensions()
    # resize_to_mean()
    # generate_negative_examples()
    # rez = np.load('./evaluare/fisiere_solutie/331_Alexe_Bogdan/task1/detections_all_faces.npy')
    # print(rez)
    pass