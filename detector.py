from copy import deepcopy
import glob
import os
import pickle
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import LinearSVC
import torch
from tqdm import tqdm
from tools import export_to_yolo, intersection_over_union, sliding_window, non_maximal_suppression, translateMap
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models.alexnet import AlexNet_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

separator = '\\' if os.name == 'nt' else '/'

base_path = os.path.join(os.getcwd(), 'data')
save_files_path = os.path.join(base_path, 'salveazaFisiere')
annotations_path = os.path.join(os.getcwd(), 'antrenare')
test_examples_path = os.path.join(os.getcwd(), 'testare')


class ImagesLoader:
    def __init__(self):
        self.loaded_images = {}
    
    def get_image(self, path):
        if self.loaded_images.get(path) is None:
            self.loaded_images[path] = cv.imread(path)
        return self.loaded_images[path]
    
    def get_image_patch(self, path, bbox):
        img = self.get_image(path)
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    


class Annotation:
    def __init__(self, bbox, label, dir, file_name):
        self.bbox = bbox
        self.label = label
        self.file_name = file_name
        self.dir = dir
        self.ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        self.shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    def __str__(self):
        return 'Annotation: bbox: %s, label: %s, dir: %s, file: %s' % (self.bbox, self.label, self.dir, self.file_name)

    def __repr__(self):
        return 'Annotation: bbox: %s, label: %s, dir: %s' % (self.bbox, self.label, self.dir, self.file_name)

    def get_file(self):
        return os.path.join(annotations_path, self.dir, self.file_name)


class AnnotationsParser:
    def __init__(self, path):
        self.path = path
        self.annotations = {}
        self.load_annotations()

    def load_annotations(self):
        annotation_files = glob.glob(os.path.join(self.path, '*.txt'))
        for file in annotation_files:
            personage_name = file.split(separator)[-1].split('_')[0]
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    file_in_dir, x1, y1, x2, y2, label = line
                    if self.annotations.get(label) is None:
                        self.annotations[label] = []
                    self.annotations[label].append(Annotation([int(x1), int(y1), int(x2), int(y2)], label, personage_name, file_in_dir))
        # for key in self.annotations:
        #     print(self.annotations[key][0].get_file())
        #     print(key, ' ', len(self.annotations[key]))

class DescriptorNNMultiClass(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def train_model(self, descriptors, labels, epochs=10, batch_size=32, lr=1e-3, device='cpu'):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        dataset = list(zip(descriptors, labels))
        for epoch in range(epochs):
            np.random.shuffle(dataset)

            epoch_loss = 0.0
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                batch_descriptors, batch_labels = zip(*batch)

                batch_descriptors = torch.tensor(batch_descriptors, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

                optimizer.zero_grad()
                outputs = self.forward(batch_descriptors)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoca [{epoch+1}/{epochs}] - Loss: {epoch_loss / len(dataset):.4f}")

    def predict(self, descriptor, device='cpu'):
        self.eval()
        with torch.no_grad():
            tensor = torch.tensor(descriptor, dtype=torch.float32).unsqueeze(0).to(device)
            rezs = self.forward(tensor)
            pred = torch.argmax(rezs, dim=1).item()
        return pred

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, device='cpu'):
        self.load_state_dict(torch.load(path))
        self.to(device)
        self.eval()


class DescriptorsNN:
    def __init__(self, embedding_dim=256):
        self.embedding_dim = embedding_dim
        self.device = device
        self.model = self.build_model()
        self.model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def train(self, descriptors, labels, epochs=10, batch_size=32):
        dataset = list(zip(descriptors, labels))
        np.random.shuffle(dataset)

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                batch_descriptors, batch_labels = zip(*batch)

                batch_descriptors = torch.tensor(batch_descriptors, dtype=torch.float32).to(self.device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_descriptors)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataset)}")

    def get_score(self, descriptor):
        self.model.eval()
        with torch.no_grad():
            descriptor = torch.tensor(descriptor, dtype=torch.float32).unsqueeze(0).to(self.device)
            score = self.model(descriptor).item()
        return score

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)


class Detection:
    def __init__(self, bboxes, scores, labels, file, yolo=False):
        self.bboxes = bboxes
        self.scores = scores
        self.labels = labels
        self.file = file
        self.yolo = yolo
    
    def str_rep(self):
        return 'Detection: bboxes: %s, scores: %s, labels: %s, file: %s' % (self.bboxes, self.scores, self.labels, self.file)
    
    def __str__(self):
        return self.str_rep()
    def __repr__(self):
        return self.str_rep()
    
    def get_all(self):
        return zip(self.bboxes, self.scores, self.labels, [self.file] * len(self.bboxes))



class FaceDetectinNN:
    def __init__(self):
        self.model_name = 'alexnet'
        self.version = '4'
        self.generate_examples = True
        self.extract_hard_negatives = False
        self.use_yolo = True

        self.detections_objs = []

        self.window_sizes = [(8 * 10, 8 * 10), (8 * 10, 8 * 6), (8 * 6, 8 * 10)]
        self.prefix = self.model_name + '_' + self.version

        self.output_path = os.path.join(base_path, 'output')
        self.parent_path_pos = os.path.join(base_path, self.prefix)
        self.positive_examples_path = os.path.join(self.parent_path_pos, 'positive')
        self.negative_examples_path = os.path.join(self.parent_path_pos, 'negative')
        self.export_hard_path = os.path.join(base_path, 'hard_examples')

        if self.generate_examples:
            try:
                self.create_examples()
            except Exception as e:
                print('Error generating examples: ', e)

        
        self.load_model('alexnet')
        # self.load_examples()
        self.load_descriptors()

        self.train_model2()
        # self.train_model()
        self.train_model_multi_class()
        self.load_yolo()
        
    def load_yolo(self):
        model = torch.hub.load("ultralytics/yolov5", "custom", path=os.path.join(base_path, self.prefix, 'best.pt'))
        model.to(device)
        
        self.yolo_model = model

    def load_model(self, model_name):
        if model_name == 'resnet18':
            from torchvision.models import resnet18
            self.model = resnet18(pretrained=True)
            self.layer = self.model._modules.get('avgpool')  
            self.embedding_dim = 512
        elif model_name == 'vgg16':
            from torchvision.models import vgg16
            self.model = vgg16(pretrained=True)
        
            self.layer = self.model.features[-1]  
            self.embedding_dim = 512
        elif model_name == 'alexnet':
            from torchvision.models import alexnet
            self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
            
            self.layer = self.model.features[-1]  
            self.embedding_dim = 256
        else:
            raise ValueError(f"Modelul {model_name} nu este suportat.")

        self.model.to(device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def create_examples(self):
        print('Generating examples')
        annotsParser = AnnotationsParser(annotations_path)
        imageLoader = ImagesLoader()

        if not os.path.exists(self.positive_examples_path):
            os.makedirs(self.positive_examples_path)
        if not os.path.exists(self.negative_examples_path):
            os.makedirs(self.negative_examples_path)

        print('path', self.positive_examples_path, self.negative_examples_path)

        
        files_pos = glob.glob(os.path.join(self.positive_examples_path, '*.jpg'))
        files_neg = glob.glob(os.path.join(self.negative_examples_path, '*.jpg'))
        if len(files_pos) > 0 and len(files_neg) > 0:
            raise ValueError('Files already exist in directories: ' + str(len(files_pos)) + ' ' + str(len(files_neg)))

        all_annotations = [ann for key in annotsParser.annotations for ann in annotsParser.annotations[key]]
        print('All annotations: ', len(all_annotations))
        #generate positive examples
        positions_in_file = {}
        for i, ann in enumerate(all_annotations):
            img = imageLoader.get_image_patch(ann.get_file(), ann.bbox)
            if positions_in_file.get(ann.get_file()) is None:
                positions_in_file[ann.get_file()] = []
            positions_in_file[ann.get_file()].append(ann.bbox)
            cv.imwrite(os.path.join(self.positive_examples_path, '%d_%s.jpg' % (i, ann.label)), img)
        
        
        negatives_needed = len(positions_in_file) // 1000 + 2
        print('Negatives needed: ', negatives_needed)


        #first save hard negatives
        hard_negatives = glob.glob(os.path.join(self.export_hard_path, '*.jpg'))
        for i, file in enumerate(hard_negatives):
            img = cv.imread(file)
            cv.imwrite(os.path.join(self.negative_examples_path, '0_0_0_%d.jpg' % i), img)

        #generate negative examples
        for key in positions_in_file:
            img = imageLoader.get_image(key)
            all_img_bboxes = positions_in_file[key]
            possible_imgs = []
            for (x, y, window, win_size) in sliding_window(img, 8 * 16, 8 * 16, self.window_sizes):
                # if window.shape != win_size:
                #     continue
                x1 = x
                y1 = y
                x2 = x + win_size[0]
                y2 = y + win_size[1]

                largest_intersection = 0
                for bbox in all_img_bboxes:
                    intersection = intersection_over_union([x1, y1, x2, y2], bbox)
                    if intersection > largest_intersection:
                        largest_intersection = intersection
                if largest_intersection < 0.3:
                    possible_imgs.append([window, largest_intersection])
            possible_imgs = sorted(possible_imgs, key=lambda x: x[1], reverse=True)
            for i, (img, _) in enumerate(possible_imgs[:negatives_needed]):
                print('saving negative example: ', i, '%d_%s.jpg' % (i, key.split(separator)[-1].split('.')[0]))
                rez = cv.imwrite(os.path.join(self.negative_examples_path, '%d_%s.jpg' % (i, key.split(separator)[-1].split('.')[0])), img)
                if rez is False:
                    print('Error saving negative example')
            

        # self.save_examples()

    def load_example_ispos(self, positive=False):
        if positive:
            path = self.positive_examples_path
        else:
            path = self.negative_examples_path

        print('Loading examples from: ', path)  

        
        files = glob.glob(os.path.join(path, '*.jpg'))
        images = [cv.imread(file) for file in files]
        images = [cv.resize(img, (224, 224)) for img in images]
        return images
    
    def load_examples(self):
        print('Loading examples')
        self.positive_examples = self.load_example_ispos(True)
        self.negative_examples = self.load_example_ispos(False)

        print('Examples loaded')
        print('Positive examples: ', len(self.positive_examples))
        print('Negative examples: ', len(self.negative_examples))

    def get_descriptor(self, img):
        img_tensor = self.preprocess(img).unsqueeze(0).to(device)
        my_embedding = torch.zeros(self.embedding_dim, device=device)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data.mean(dim=[2, 3]).view(-1))

        h = self.layer.register_forward_hook(copy_data)
        self.model(img_tensor)
        h.remove()

        return my_embedding.cpu().numpy()
    
    def load_descriptors_ispos(self, positive=True):
        prefix = 'positive' if positive else 'negative'
        file_name = prefix + 'descriptors.npy'
        file_path = os.path.join(self.parent_path_pos, file_name)
        if os.path.exists(file_path):
            if positive:
                self.positive_descriptors = np.load(file_path)
            else:
                self.negative_descriptors = np.load(file_path)
            return
        
        examples = self.positive_examples if positive else self.negative_examples

        descriptors = []
        for img in tqdm(examples, desc="Processing positive examples" if positive else "Processing negative examples"):
            descriptor = self.get_descriptor(img)
            descriptors.append(descriptor)
        
        if positive:
            self.positive_descriptors = np.array(descriptors)
            np.save(file_path, self.positive_descriptors)
        else:
            self.negative_descriptors = np.array(descriptors)
            np.save(file_path, self.negative_descriptors)
    
    def load_personages_descriptor(self):
        base_path = os.path.join(os.getcwd(), 'data', self.prefix)
        personages = ['dad', 'mom', 'deedee', 'dexter', 'unknown']
        personages_descriptors = {}
        annots = None
        for personage in personages:
            personage_file = os.path.join(base_path, personage + '_descriptors.npy')
            print(personage_file)
            if os.path.exists(personage_file):
                personages_descriptors[personage] = np.load(personage_file)
            else:
                if not annots:
                    annots = AnnotationsParser(annotations_path)
                personage_annots = annots.annotations[personage]
                personage_descriptors = []
                for ann in tqdm(personage_annots, desc=f"Processing {personage} examples"):
                    img = cv.imread(ann.get_file())
                    img = img[ann.bbox[1]:ann.bbox[3], ann.bbox[0]:ann.bbox[2]]
                    img = cv.resize(img, (224, 224))
                    descriptor = self.get_descriptor(img)
                    personage_descriptors.append(descriptor)
                    if personage == 'unknown':
                        img_aug = np.fliplr(img)
                        descriptor = self.get_descriptor(img_aug)
                        personage_descriptors.append(descriptor)
                        for i in range(1, 2):
                            img_rot = cv.rotate(img, cv.ROTATE_180)
                            descriptor = self.get_descriptor(img_rot)
                            personage_descriptors.append(descriptor)
                personages_descriptors[personage] = np.array(personage_descriptors)
                np.save(personage_file, personages_descriptors[personage])
        self.personages_descriptors = personages_descriptors
            

    def load_descriptors(self):
        print('Loading descriptors')
        self.load_descriptors_ispos(True)
        self.load_descriptors_ispos(False)
        self.load_personages_descriptor()

        print('Descriptors loaded')
        print('Positive descriptors: ', len(self.positive_descriptors))
        print('Negative descriptors: ', len(self.negative_descriptors))

        for personage in self.personages_descriptors:
            print(personage, ' descriptors: ', len(self.personages_descriptors[personage]))

    def train_model_multi_class(self):
        model_path = os.path.join(self.parent_path_pos, 'multi_class_model_nn')
        if os.path.exists(model_path):
            self.multi_class_model = DescriptorNNMultiClass(self.embedding_dim, 5)
            self.multi_class_model.load_model(model_path, device)
            return
        unknown_descriptors = self.personages_descriptors['unknown']
        mom_descriptors = self.personages_descriptors['mom']
        dad_descriptors = self.personages_descriptors['dad']
        dexter_descriptors = self.personages_descriptors['dexter']
        deedee_descriptors = self.personages_descriptors['deedee']
        min_len = min(len(mom_descriptors), len(dad_descriptors), len(dexter_descriptors), len(deedee_descriptors), len(unknown_descriptors))
        mom_descriptors = mom_descriptors[:min_len]
        dad_descriptors = dad_descriptors[:min_len]
        dexter_descriptors = dexter_descriptors[:min_len]
        deedee_descriptors = deedee_descriptors[:min_len]
        unknown_descriptors = unknown_descriptors[:min_len]
        print('Min len: ', min_len)


        descs = np.concatenate([
            mom_descriptors,
            dad_descriptors,
            dexter_descriptors,
            deedee_descriptors,
            unknown_descriptors
        ], axis=0)

        labels = np.concatenate([
            np.full(len(mom_descriptors), 0),
            np.full(len(dad_descriptors), 1),
            np.full(len(dexter_descriptors), 2),
            np.full(len(deedee_descriptors), 3),
            np.full(len(unknown_descriptors), 4)
        ], axis=0)

        model = DescriptorNNMultiClass(self.embedding_dim, 5)
        model.train_model(descs, labels, 10, 32, 1e-3, device)
        model.save_model(model_path)
        self.multi_class_model = model


    def train_model2(self):
        path_to_model = os.path.join(self.parent_path_pos, 'best_model_nn')
        if os.path.exists(path_to_model):
            self.nn_classifier = DescriptorsNN(self.embedding_dim)
            self.nn_classifier.load_model(path_to_model)
            return
        
        if len(self.positive_descriptors) != len(self.negative_descriptors):
            min_len = min(len(self.positive_descriptors), len(self.negative_descriptors))
            self.positive_descriptors = self.positive_descriptors[:min_len]
            self.negative_descriptors = self.negative_descriptors[:min_len]

        train_descriptors = np.concatenate((np.squeeze(self.positive_descriptors), np.squeeze(self.negative_descriptors)), axis=0)
        train_labels = np.concatenate((np.ones(len(self.positive_descriptors)), np.zeros(len(self.negative_descriptors))))

        self.nn_classifier = DescriptorsNN(self.embedding_dim)
        self.nn_classifier.train(train_descriptors, train_labels, 10)

        self.nn_classifier.save_model(path_to_model)

        print("Training completed.")

    def train_model(self, Cs=None):
        svm_file_name = os.path.join(self.parent_path_pos, 'best_model')
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return
        

        if Cs is None:
            Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]

        best_accuracy = 0
        best_c = 0
        best_model = None

        if len(self.positive_descriptors) != len(self.negative_descriptors):
            min_len = min(len(self.positive_descriptors), len(self.negative_descriptors))
            self.positive_descriptors = self.positive_descriptors[:min_len]
            self.negative_descriptors = self.negative_descriptors[:min_len]

        train_descriptors = np.concatenate((np.squeeze(self.positive_descriptors), np.squeeze(self.negative_descriptors)), axis=0)
        train_labels = np.concatenate((np.ones(len(self.positive_descriptors)), np.zeros(len(self.negative_descriptors))))

        for c in Cs:
            print(f'Antrenăm un clasificator pentru C={c}')
            model = LinearSVC(C=c)
            model.fit(train_descriptors, train_labels)
            acc = model.score(train_descriptors, train_labels)
            print(f'Performanța clasificatorului pentru C={c}: {acc}')
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print(f'Performanța clasificatorului optim pentru C={best_c}: {best_accuracy}')
        
        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(train_descriptors)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.savefig(os.path.join(self.parent_path_pos, 'distributie.png'))
        plt.show()

    def get_task_npy_files(self, yolo=False, label=None):
        detections = []
        scores = []
        labels = []
        files = []

        for det_obj in self.detections_objs:
            if (det_obj.yolo and not yolo) or (not det_obj.yolo and yolo):
                continue
            for det_tuple in tuple(det_obj.get_all()):
                det, score, det_label, file = det_tuple
                if label is not None and det_label != label:
                    continue
                detections.append(det)
                scores.append(score)
                files.append(file)
        
        return np.array(detections), np.array(scores), np.array(files)

    # def save_temp_detctions(self):

    def save_files(self, detections, scores, files, label=None, yolo=False, temp=False):
        task = 'task1' if label is None else 'task2'
        if yolo:
            task = f'{task}_yolo'
        if temp:
            task = f'{task}_temp'
        base_path = os.path.join(self.output_path, task)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        label_map = {
            0: 'mom',
            1: 'dad',
            2: 'dexter',
            3: 'deedee',
        }
        label_name = label_map.get(label, None)
        if label_name is None and label is not None:
            raise ValueError('Label not found: ', label)
        dets_file_name = 'detections_all_faces.npy' if label is None else f'detections_{label_name}.npy'
        scores_file_name = 'scores_all_faces.npy' if label is None else f'scores_{label_name}.npy'
        files_file_name = 'file_names_all_faces.npy' if label is None else f'file_names_{label_name}.npy'
        
        np.save(os.path.join(base_path, dets_file_name), detections)
        np.save(os.path.join(base_path, scores_file_name), scores)
        np.save(os.path.join(base_path, files_file_name), files)

    def save_detections(self, temp=False):
        def save_by_yolo(yolo=False):
            #TASK 1
            detections, scores, files = self.get_task_npy_files(yolo=yolo)
            self.save_files(detections, scores, files, yolo=yolo, temp=temp)

            #TASK 2
            #for every label
            for label in [0, 1, 2, 3]:
                detections, scores, files = self.get_task_npy_files(yolo=yolo, label=label)
                self.save_files(detections, scores, files, label, yolo=yolo, temp=temp)

        #BASE solution
        save_by_yolo(yolo=False)

        # YOLO solution
        save_by_yolo(yolo=True)


    def run(self):
        files_to_detect = glob.glob(os.path.join(test_examples_path, '*.jpg'))
        files_to_detect = sorted(files_to_detect)
        self.detections_objs = []

        if self.extract_hard_negatives:
            if not os.path.exists(self.export_hard_path):
                os.makedirs(self.export_hard_path)
            annots_dict = {}
            annots = AnnotationsParser(annotations_path)
            for key in ['deedee', 'dexter', 'unknown', 'dad', 'mom']:
                for ann in annots.annotations[key]:
                    if annots_dict.get(ann.get_file()) is None:
                        annots_dict[ann.get_file()] = []
                    annots_dict[ann.get_file()].append(ann)
            files_to_detect = list(annots_dict.keys())

        print('Files to detect: ', len(files_to_detect))

        for counter, file in enumerate(files_to_detect):
            try:
                # if counter < 2:
                #     continue
                img = cv.imread(file)
                original_shape = img.shape
                print('#' * 50)
                print('Detecting for image: ', file)
                print('Original shape: ', original_shape)
                detection_obj_yolo = Detection([], [], [], os.path.basename(file), True)
                dection_obj_base = Detection([], [], [], os.path.basename(file), False)
                ################# YOLO #################
                if self.use_yolo:
                    # O ZI PIERDUTA REANTRENAND DE CATEVA ORI MODELUL DE YOLO
                    # PENTRU CA AVEA ACURATETEA SCAZUTA IN SOLUTIE INAINTE DE A
                    # IMI DA SEAMA CA OPENCV ESTE BGR IAR YOLO E ANTRENAT PE RGB
                    yolo_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    results = self.yolo_model(yolo_img)

                    results = results.xyxy[0].cpu().numpy()
                    for i, res in enumerate(results):
                        x1, y1, x2, y2, score, label = res
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        detection_obj_yolo.bboxes.append([x1, y1, x2, y2])
                        detection_obj_yolo.scores.append(score)
                        detection_obj_yolo.labels.append(label)
                    self.detections_objs.append(detection_obj_yolo)
                    # continue
                ########################################

                ################# BASE #################
                detections = []
                scores = []
                for window_size in self.window_sizes:
                    scale = 1.0
                    img_scaled = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
                    
                    while img_scaled.shape[0] >= window_size[1] and img_scaled.shape[1] >= window_size[0]:
                        for (x, y, window, win_size) in sliding_window(img_scaled, img.shape[1] // 40, img.shape[0] // 40, [window_size]):
                            x1 = x
                            y1 = y
                            x2 = x + win_size[0]
                            y2 = y + win_size[1]

                            window = cv.resize(window, (224, 224))
                            descriptor = self.get_descriptor(window)
                            # score = self.best_model.decision_function([descriptor])
                            score = self.nn_classifier.get_score(descriptor)
                            if score > 0.8:
                                x1 = int(x1 / scale)
                                y1 = int(y1 / scale)
                                x2 = int(x2 / scale)
                                y2 = int(y2 / scale)

                                detections.append([x1, y1, x2, y2])
                                scores.append(score)
                                
                        scale *= 0.8
                        img_scaled = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

                if len(detections) == 0:
                    print('No detections for image: ', file)
                    continue
                images_det, scores_det = non_maximal_suppression(
                    np.array(detections), np.array(scores).flatten(), original_shape, 0.1)

                if self.extract_hard_negatives:
                    # HARD MINING
                    real_bboxes = [ann.bbox for ann in annots_dict[file]]
                    good_detected = 0
                    for i, (det, score) in enumerate(zip(images_det, scores_det)):
                        x1, y1, x2, y2 = det
                        ious = [intersection_over_union(det, bbox) for bbox in real_bboxes]
                        max_iou = max(ious)
                        if max_iou < 0.3:
                            cv.imwrite(os.path.join(self.export_hard_path, 'hard_%d_%d_%s.jpg' % (counter, i, file.split(separator)[-1].split('.')[0])), img[y1:y2, x1:x2])
                            good_detected -= 1
                        else:
                            good_detected += 1
                    print('Good detections: ', good_detected, '/', len(real_bboxes))
                    continue

                for i, (det, score) in enumerate(zip(images_det, scores_det)):
                    if score < 0.9 and i > 1:
                        continue
                    print('Detected: ', det, ' with score: ', score)
                    x1, y1, x2, y2 = det
                    h, w = img.shape[:2]
                    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
                    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
                    print('Real shape: ', img.shape)
                    print(x1, y1, x2, y2)
                    window = img[y1:y2, x1:x2]
                    print('Window shape: ', window.shape)
                    img_crop = cv.resize(window, (224, 224))
                    descr = self.get_descriptor(img_crop)
                    pred = self.multi_class_model.predict(descr, device)
                    print('Predicted: ', pred)

                    real_x1 = x1
                    real_y1 = y1
                    real_x2 = x2
                    real_y2 = y2

                    dection_obj_base.bboxes.append([real_x1, real_y1, real_x2, real_y2])
                    dection_obj_base.scores.append(score)
                    dection_obj_base.labels.append(pred)


                    # color = (0, int(translateMap(score, 0.8, 1.0, 0, 255)), 0)
                    # cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    # cv.putText(img, str(score), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
                self.detections_objs.append(dection_obj_base)
                self.save_detections(temp=True)
                
                # cv.imshow('Detected', img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
            except Exception as e:
                print('Error detecting for file: ', file, ' error: ', e)

        self.save_detections()
    


if __name__ == '__main__':
    face_detection = FaceDetectinNN()
    face_detection.run()


# VERSIUNEA ANTERIOARA CU HOG CARE NU A DAT REZULTATE PREA BUNE
# class FaceDetection:
#     def __init__(self, dim_hog_cell=8,
#                  overlap=0.3, flip_images=True):
#         self.dim_hog_cell = dim_hog_cell

#         self.dim_windows = [(8 * 16, 8 * 16)]
#         self.best_model = {}

#         for window_dims in self.dim_windows:
#             if window_dims[0] % self.dim_hog_cell != 0 or window_dims[1] % self.dim_hog_cell != 0:
#                 raise ValueError('Dimensiunile ferestrei trebuie sa fie multiplu de dimensiunea celulei HOG')

#         if os.path.exists(save_files_path) is False:
#             os.makedirs(save_files_path)
#             print('directory created: {} '.format(save_files_path))

#         self.overlap = overlap
#         self.positive_descriptors = {}
#         self.negative_descriptors = {}
#         self.flip_images = flip_images

#         self.annotations = {
#             'dad': self.get_annotations('dad'),
#             'mom': self.get_annotations('mom'),
#             'deedee': self.get_annotations('deedee'),
#             'dexter': self.get_annotations('dexter')
#         }

#         print('Annotations loaded')
#         for key in self.annotations:
#             print(key, ' ', len(self.annotations[key]))

#     def get_annotations(self, personage):
#         annotations = {}
#         with open(os.path.join(annotations_path, personage + '_annotations.txt'), 'r') as f:
#             for line in f:
#                 line = line.strip().split(' ')
#                 if annotations.get(line[0]) is None:
#                     annotations[line[0]] = []
#                 annotations[line[0]].append([int(line[1]), int(line[2]), int(line[3]), int(line[4])])
#         return annotations

#     def get_files_like(self, path, rule, max_files=None, return_string_list=False):
#         if not os.path.exists(path):
#             os.makedirs(path)
#         images_path = os.path.join(path, rule)
#         dad = 0
#         mom = 0
#         deedee = 0
#         dexter = 0
#         unknown = 0

#         files = glob.glob(images_path)
#         for file in files:
#             if 'dad' in file:
#                 dad += 1
#             if 'mom' in file:
#                 mom += 1
#             if 'deedee' in file:
#                 deedee += 1
#             if 'dexter' in file:
#                 dexter += 1
#             if 'unknown' in file:
#                 unknown += 1
#         print('dad: ', dad, ' mom: ', mom, ' deedee: ', deedee, ' dexter: ', dexter, ' unknown: ', unknown)

#         files = sorted(files)

#         if max_files is not None:
#             files = files[:max_files]

#         files_read = [cv.imread(file) for file in files]
#         if not return_string_list:
#             return files_read
#         return files_read, files


#     def load_examples(self):
#         print('Loading examples')
#         self.positive_examples = self.get_files_like(positive_examples_path, '*.jpg')
#         local_neg, filenames = self.get_files_like(negative_examples_path, '*.jpg', return_string_list=True)
#         if self.flip_images:
#             self.positive_examples += [np.fliplr(img) for img in self.positive_examples]
#         # for i in range(1, 4):
#         #     self.positive_examples += [cv.rotate(img, i) for img in self.positive_examples]
#         # for i in range(len(local_neg)):
#         #     imshow_no_display(local_neg[i], filenames[i].split('\\')[-1])
#         self.negative_examples = (local_neg, filenames)

#         # top_dad = self.get_files_like(negative_examples_path, 'top_dad_*.jpg')
#         # top_mom = self.get_files_like(negative_examples_path, 'top_mom_*.jpg')
#         # top_deedee = self.get_files_like(negative_examples_path, 'top_deedee_*.jpg')
#         # top_dexter = self.get_files_like(negative_examples_path, 'top_dexter_*.jpg')

#         # self.negative_examples = top_dad[:2000] + top_mom[:2000] + top_deedee[:2000] + top_dexter[:2000]

#         # hard_dad = self.get_files_like('./hard_dad', '*.jpg')
#         # hard_mom = self.get_files_like('./hard_mom', '*.jpg')
#         # hard_deedee = self.get_files_like('./hard_deedee', '*.jpg')
#         # hard_dexter = self.get_files_like('./hard_dexter', '*.jpg')

#         # all_hard = hard_dad + hard_mom + hard_deedee + hard_dexter

#         # all_hard += [np.fliplr(img) for img in all_hard]

#         # print('All hard_picked: ', len(all_hard))
#         # self.negative_examples = all_hard

#         # negative_needed = len(self.positive_examples) - len(self.negative_examples)
#         # negative_needed = negative_needed // 4 + 1

#         # print('Negative additive needed: ', negative_needed)

#         # self.negative_examples += top_dad[:negative_needed] + top_mom[:negative_needed] + top_deedee[:negative_needed] + top_dexter[:negative_needed]

#         print('Negative examples total: ', len(self.negative_examples[0]))


#         # for i in range(1, 4):
#         #     self.positive_examples += [cv.rotate(img, i) for img in self.positive_examples]

#         # min_len = min(len(self.positive_examples), len(self.negative_examples))
#         # self.positive_examples = self.positive_examples[:min_len]
#         # self.negative_examples = self.negative_examples[:min_len]

#         print('Examples loaded: ', len(self.positive_examples), len(self.negative_examples[0]))
    
#     def window_dim_to_str(self, window_dim):
#         return '%d_%d' % (window_dim[0], window_dim[1])


#     def get_descriptors_isneg(self, is_negative=False):
#         if is_negative:
#             # init_examples = self.negative_examples[0]
#             needed_examples = len(self.positive_examples)
#             equal_number = needed_examples // 4 + 1
#             dad_examples = []
#             mom_examples = []
#             deedee_examples = []
#             dexter_examples = []
#             for i in range(len(self.negative_examples[0])):
#                 if 'dad' in self.negative_examples[1][i]:
#                     dad_examples.append(self.negative_examples[0][i])
#                 if 'mom' in self.negative_examples[1][i]:
#                     mom_examples.append(self.negative_examples[0][i])
#                 if 'deedee' in self.negative_examples[1][i]:
#                     deedee_examples.append(self.negative_examples[0][i])
#                 if 'dexter' in self.negative_examples[1][i]:
#                     dexter_examples.append(self.negative_examples[0][i])
                    
#             print('Dad examples: ', len(dad_examples))
#             print('Mom examples: ', len(mom_examples))
#             print('Deedee examples: ', len(deedee_examples))
#             print('Dexter examples: ', len(dexter_examples))
#         else:
#             init_examples = self.positive_examples

#         for window_dims in self.dim_windows:
#             str_window = self.window_dim_to_str(window_dims)
#             if is_negative:
#                 filtered_dad = [img for img in dad_examples if img.shape[0] >= window_dims[1] and img.shape[1] >= window_dims[0]]
#                 filtered_mom = [img for img in mom_examples if img.shape[0] >= window_dims[1] and img.shape[1] >= window_dims[0]]
#                 filtered_deedee = [img for img in deedee_examples if img.shape[0] >= window_dims[1] and img.shape[1] >= window_dims[0]]
#                 filtered_dexter = [img for img in dexter_examples if img.shape[0] >= window_dims[1] and img.shape[1] >= window_dims[0]]
#                 examples = filtered_dad[:equal_number] + filtered_mom[:equal_number] + filtered_deedee[:equal_number] + filtered_dexter[:equal_number]
#                 hard_dad = self.get_files_like('./show', 'dad_*.jpg')
#                 print('Hard dad: ', len(hard_dad))
#                 examples = hard_dad + examples

#                 shapes = [img.shape for img in examples]
#                 shapes = set(shapes)
#                 print('Shapes: ', shapes)
#                 print('Negative examples for window: ', str_window, ' ', len(examples))
#                 # examples = [img for img in init_examples if img.shape[0] == window_dims[1] and img.shape[1] == window_dims[0]]
#             else:
#                 examples = [cv.resize(img, (window_dims[0], window_dims[1])) for img in init_examples]
#             print('Window: ', str_window, ' Examples: ', len(examples))
#             file_name = '%d_%d_descriptoriExemplePozitive1_%d_%d.npy' % (window_dims[0], window_dims[1], self.dim_hog_cell, len(examples)) if not is_negative else '%d_%d_descriptoriExempleNegative1_%d_%d.npy' % (window_dims[0], window_dims[1], self.dim_hog_cell, len(examples))
#             npyPath = os.path.join(save_files_path, file_name)
#             if os.path.exists(npyPath):
#                 if is_negative:
#                     self.negative_descriptors[str_window] = np.load(npyPath)
#                 else:
#                     self.positive_descriptors[str_window] = np.load(npyPath)
#                 continue

#             descriptors = []
#             show_train_path = './show_train/' + str_window
#             if not os.path.exists(show_train_path):
#                 os.makedirs(show_train_path)
#             for img in tqdm(examples, desc="Processing negative examples" if is_negative else "Processing positive examples"):
#                 img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#                 cv.imwrite(os.path.join(show_train_path, '%d_%s_img.jpg' % (len(descriptors), "neg" if is_negative else "poz")), img_gray)
#                 features, vis = hog(
#                     img_gray,
#                     pixels_per_cell=(self.dim_hog_cell, self.dim_hog_cell),
#                     cells_per_block=(2, 2),
#                     feature_vector=True,
#                     visualize=True,
#                 )
#                 cv.imwrite(os.path.join(show_train_path, '%d_%s_hog.jpg' % (len(descriptors), "neg" if is_negative else "poz")), vis)
#                 descriptors.append(features)
#             print('Npypath: ', npyPath)
#             if is_negative:
#                 self.negative_descriptors[str_window] = np.array(descriptors)
#                 np.save(npyPath, self.negative_descriptors[str_window])
#             else:
#                 self.positive_descriptors[str_window] = np.array(descriptors)
#                 np.save(npyPath, self.positive_descriptors[str_window])

#         if is_negative:
#             print('Negative descriptors loaded')
#             return self.negative_descriptors
#         else:
#             print('Positive descriptors loaded')
#             return self.positive_descriptors
            

#     def get_descriptors(self):
#         print('Getting descriptors')
#         self.get_descriptors_isneg(True)
#         self.get_descriptors_isneg(False)

#         print('Descriptors loaded')
#         print('Positive descriptors: ', len(self.positive_descriptors))
#         print('Negative descriptors: ', len(self.negative_descriptors))

#     def train_classifier(self):
#         print('Training classifier')
#         for window_dims in self.dim_windows:
#             string_window = self.window_dim_to_str(window_dims)
#             svm_file_name = os.path.join(save_files_path, '%d_%d_%s_best_model_%d_%d_%d' %
#                                         (window_dims[0], window_dims[1], 'hogv2', self.dim_hog_cell, self.positive_descriptors[string_window].shape[0],
#                                         self.negative_descriptors[string_window].shape[0]))
#             if os.path.exists(svm_file_name):
#                 self.best_model[string_window] = pickle.load(open(svm_file_name, 'rb'))
#                 continue
            
#             local_positive_descriptors = self.positive_descriptors[string_window]
#             local_negative_descriptors = self.negative_descriptors[string_window]

#             print(local_negative_descriptors[0].shape)
#             print(local_positive_descriptors[0].shape)

#             print('Training classifier for window: ', string_window)
#             print('Positive descriptors: ', len(local_positive_descriptors))
#             print('Negative descriptors: ', len(local_negative_descriptors))

#             min_len = min(len(local_positive_descriptors), len(local_negative_descriptors))

#             print(len(local_positive_descriptors))
#             print(len(local_positive_descriptors))

#             local_negative_descriptors = local_negative_descriptors[:min_len]
#             local_positive_descriptors = local_positive_descriptors[:min_len]

#             training_examples = np.concatenate((np.squeeze(local_positive_descriptors), np.squeeze(local_negative_descriptors)), axis=0)
#             train_labels = np.concatenate((np.ones(len(local_positive_descriptors)), np.zeros(len(local_negative_descriptors))))

#             best_accuracy = 0
#             best_c = 0
#             best_model = None
#             Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1]
#             for c in Cs:
#                 print('Antrenam un clasificator %s cu c = %f' % (string_window, c))
#                 model = LinearSVC(C=c)
#                 model.fit(training_examples, train_labels)
#                 acc = model.score(training_examples, train_labels)
#                 print(acc)
#                 if acc > best_accuracy:
#                     best_accuracy = acc
#                     best_c = c
#                     best_model = deepcopy(model)

#             print('Performanta clasificatorului optim pt c = %f' % best_c)
#             # salveaza clasificatorul
#             pickle.dump(best_model, open(svm_file_name, 'wb'))

#             # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
#             # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
#             scores = best_model.decision_function(training_examples)
#             self.best_model[string_window] = best_model
#             positive_scores = scores[train_labels > 0]
#             negative_scores = scores[train_labels <= 0]

#             plt.clf()
#             plt.plot(np.sort(positive_scores))
#             plt.plot(np.zeros(len(positive_scores)))
#             plt.plot(np.sort(negative_scores))
#             plt.xlabel('Nr example antrenare')
#             plt.ylabel('Scor clasificator')
#             plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
#             plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
#             plt.savefig(os.path.join(save_files_path, '%d_%d_distributie_%d_%d_%d.png' % (window_dims[0], window_dims[1], self.dim_hog_cell, self.positive_descriptors[string_window].shape[0],
#                                         self.negative_descriptors[string_window].shape[0])))


#     def run(self):
#         # test_files, string_list = self.get_files_like(test_examples_path, '*.jpg', return_string_list=True)

#         # print(string_list)
#         test_files, string_list = self.get_files_like('./antrenare/dad', '*.jpg', return_string_list=True)
#         #shuffle test files

#         npy_detections_list = []
#         npy_scores_list = np.array([])
#         npy_file_names_list = np.array([])

#         w= {}
#         bias = {}
#         score_thresh = {}
#         for window_dims in self.dim_windows:
#             str_window = self.window_dim_to_str(window_dims)
#             w[str_window] = self.best_model[str_window].coef_[0]
#             bias[str_window] = self.best_model[str_window].intercept_[0]
#             # if str_window is self.window_dim_to_str(self.dim_windows[0]):
#             score_thresh[str_window] = 3
#             # elif str_window is self.window_dim_to_str(self.dim_windows[1]):
#             #     score_thresh[str_window] = 5
#             # else:
#             #     score_thresh[str_window] = 5

        
#         for counter, img in enumerate(test_files):
#             img = cv.resize(img, (img.shape[1] * 4, img.shape[0] * 4))
#             img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#             original_shape = img_gray.shape
#             print('Processing image: ', string_list[counter])
#             print('shape: ', img_gray.shape)

#             detections = []
#             scores = []

#             scale_factor = 0.8
#             for window_size in self.dim_windows:
#                 scale = 1.0
#                 img_gray_scaled = cv.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
#                 str_window = self.window_dim_to_str(window_size)
#                 print('current window size: ', window_size)
#                 while img_gray_scaled.shape[0] >= window_size[1] and img_gray_scaled.shape[1] >= window_size[0]:
#                     features_all, vis = hog(
#                                         img_gray_scaled,
#                                         pixels_per_cell=(self.dim_hog_cell, self.dim_hog_cell),
#                                         cells_per_block=(2, 2),
#                                         feature_vector=False,
#                                         visualize=True,
#                                 )

#                     imshow_no_display(vis, 'hog_%d_%f' % (counter, scale))


#                     num_cols = img_gray_scaled.shape[1] // self.dim_hog_cell - 1
#                     num_rows = img_gray_scaled.shape[0] // self.dim_hog_cell - 1
#                     num_cell_in_template_h = window_size[1] // self.dim_hog_cell - 1
#                     num_cell_in_template_w = window_size[0] // self.dim_hog_cell - 1
                    
#                     for y in range(0, num_rows - num_cell_in_template_h):
#                         for x in range(0, num_cols - num_cell_in_template_w):
#                             descr = features_all[y:y + num_cell_in_template_h, x:x + num_cell_in_template_w].flatten()
#                             score = np.dot(descr, w[str_window]) + bias[str_window]
#                             if score > score_thresh[str_window]:
#                                 x_min = int(x * self.dim_hog_cell)
#                                 y_min = int(y * self.dim_hog_cell)
#                                 x_max = int(x * self.dim_hog_cell + window_size[0])
#                                 y_max = int(y * self.dim_hog_cell + window_size[1])


#                                 x_min_scaled = int(x_min / scale)
#                                 y_min_scaled = int(y_min / scale)
#                                 x_max_scaled = int(x_max / scale)
#                                 y_max_scaled = int(y_max / scale)

#                                 # if detections.get(str_window) is None:
#                                 #     detections[str_window] = []
#                                 #     scores[str_window] = []
                                
#                                 detections.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])
#                                 scores.append(score)

#                                 # imshow_no_display(img_gray[y_min:y_max, x_min:x_max], '%d_%d_%d_window' % (counter, x, y))
#                                 # imshow_no_display(vis[y_min:y_max, x_min:x_max], '%d_%d_%d_hog' % (counter, x, y))
#                     scale *= scale_factor
#                     img_gray_scaled = cv.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

#             det_len = len(detections)
#             if det_len > 0:
#                 # for str_window in detections:
#                 # print('Window: ', str_window, ' Detections: ', len(detections[str_window]))
#                 image_detections, image_scores = non_maximal_suppression(
#                         np.array(detections), np.array(scores), original_shape, 0.5)
    
#                 real_annotations = self.annotations['dad'].get(string_list[counter].split('\\')[-1], [])
#                 for real_ann in real_annotations:
#                     real_ann = [real_ann[0] * 4, real_ann[1] * 4, real_ann[2] * 4, real_ann[3] * 4]
#                     x1, y1, x2, y2 = real_ann
#                     cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


#                 # filter detections that have nms lower than 0.3
#                 for i, det in enumerate(image_detections):
#                     # is_good = True
#                     # for real_ann in real_annotations:
#                     #     real_ann = [real_ann[0] * 4, real_ann[1] * 4, real_ann[2] * 4, real_ann[3] * 4]
#                     #     if intersection_over_union(det, real_ann) > 0.3:
#                     #         is_good = False
#                     #         break
#                     # if not is_good:
#                     #     continue

#                     # if image_scores[i] < 1 and i > 0:
#                     #     print('Score too low: ', det, ' Score: ', image_scores[i])
#                     #     x1, y1, x2, y2 = det
#                     #     cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     #     cv.putText(img, str(image_scores[i]), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                     #     continue
#                     print('Det: ', det, ' Score: ', image_scores[i])
#                     # if image_scores[i] > 3:
#                     #     det_img_resized = cv.resize(img[det[1]:det[3], det[0]:det[2]], (8 * 16, 8 * 16))
#                     #     imshow_no_display(det_img_resized, 'dad_%f_%d_%d_%d_window' % (image_scores[i], counter, det[0], det[1]))
#                     # npy_detections_list.append(det)
#                     # print(npy_detections_list)
#                     # npy_scores_list = np.append(npy_scores_list, image_scores[i])
#                     # npy_file_names_list = np.append(npy_file_names_list, string_list[counter].split('\\')[-1])
#                     x1, y1, x2, y2 = det
#                     color = (255, 0, 0)
#                     # if str_window is self.window_dim_to_str(self.dim_windows[0]):
#                     #     color = (0, 255, 0)
#                     # elif str_window is self.window_dim_to_str(self.dim_windows[1]):
#                     #     color = (0, 0, 255)
#                     cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
#                     cv.putText(img, str(image_scores[i]), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                 imshow_no_display(img, 'mom_%d' % counter)
#             else:
#                 print('No detections for image: ', string_list[counter])

        
#         np.save(os.path.join(test_gt_path, 'fin_detections_all_faces.npy'), np.array(npy_detections_list))
#         np.save(os.path.join(test_gt_path, 'fin_scores_all_faces.npy'), npy_scores_list)
#         np.save(os.path.join(test_gt_path, 'fin_file_names_all_faces.npy'), npy_file_names_list)





# faceDet = FaceDetection()
# faceDet.load_examples()
# faceDet.get_descriptors()
# faceDet.train_classifier()
# faceDet.run()