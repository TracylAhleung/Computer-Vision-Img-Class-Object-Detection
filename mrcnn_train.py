## MASK_RCNN SET UP
# Step 1: Clone the repository onto your command window
# Step 2: Import the required libraries
# Step 3: Download MS COCO (using a pre-trained model for now)         check requirements.txt for package versions
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import utils
import tensorflow
import pixellib
import cv2
import mrcnn.config
import mrcnn.model as modellib
import xml.etree
from xml.etree import ElementTree
from pixellib.instance import instance_segmentation
from mrcnn import utils
from numpy import zeros, asarray
from mrcnn import visualize

# Root directory of the project
dataset_dir = os.path.abspath("C:/Users/tracy/PycharmProjects/ImageClassification/Sunflower pic/")
# Import Mask RCNN
sys.path.append(dataset_dir)  # To find local version of the library
#
#
# # Preparing the model configuration parameters
# class SimpleConfig(mrcnn.config.Config):
#     NAME = "COCO_INFERENCE"
#     NUM_CLASSES = 81  # 80 classes in the COCO dataset + background(1)
#     IMAGES_PER_GPU = 1  # INPUT 1 SINGLE IMAGE FOR NOW
#     GPU_COUNT = 1
#     BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT
#
#
# # BUILDING THE MRCNN ARCHITECTURE
# def mrcnn_model():
#     model = mrcnn.model.MaskRCNN(mode="inference",
#                                  config=SimpleConfig(),
#                                  model_dir=os.getcwd())
#     model.keras_model.summary()
#     model.load_weights(filepath="mask_rcnn_coco.h5",
#                        by_name=True)
#
#     image = cv2.imread("3b0b85b5df9b9b0f8dde45c6b59b8e2560359373.jpg")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     r = model.detect(images=[image],
#                      verbose=0)
#
#     CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
#                    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#                    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
#                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#                    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#
#     r = r[0]
#     print(r.keys())
#     mrcnn.visualize.display_instances(image=image,
#                                       boxes=r['rois'],
#                                       masks=r['masks'],
#                                       class_ids=r['class_ids'],
#                                       class_names=CLASS_NAMES,
#                                       scores=r['scores'])
#     return




# Setup the model
# segmentation_model = instance_segmentation()
# segmentation_model.load_model("mask_rcnn_coco.h5")


# def real_time_Capture():
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret,frame = cap.read()
#
#         #Apply instance segmentation
#         res=segmentation_model.segmentFrame(frame,show_bboxes=True)
#         image=res[1]
#         cv2.imshow('Instance Segmentation',image)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#     return


class FlowerDataset(mrcnn.utils.Dataset):
    dataset_dir = "C:/Users/tracy/PycharmProjects/ImageClassification/Sunflower pic/"
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "sunflower")
        self.add_class("dataset", 2, "stigma")
        self.add_class("dataset", 3, "b_sunflower")

        images_dir = "C:/Users/tracy/PycharmProjects/ImageClassification/Sunflower pic/Images/"
        annotations_dir = "C:/Users/tracy/PycharmProjects/ImageClassification/Sunflower pic/annotations/"

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            #add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2,3])

    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            if name == 'sunflower' or name=='stigma' or name=="b_sunflower":
                boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if (box[4] == 'sunflower'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('sunflower'))
            elif (box[4] =='stigma'):
                masks[row_s:row_e, col_s:col_e,i] = 2
                class_ids.append(self.class_names.index('stigma'))
            else:
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('b_sunflower'))

        return masks, np.array(class_ids).astype(np.uint8)


class flowerConfig(mrcnn.config.Config):
    NAME = "flower_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1+3

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.90


#Directory
valid_dir ="C:/Users/tracy/PycharmProjects/ImageClassification/Sunflower pic/Validate/"
train_dir = "C:/Users/tracy/PycharmProjects/ImageClassification/Sunflower pic/Part 1/"

train_set = FlowerDataset()
train_set.load_dataset(dataset_dir=train_dir, is_train=True)
train_set.prepare()


valid_dataset = FlowerDataset()
valid_dataset.load_dataset(dataset_dir=valid_dir, is_train=False)
valid_dataset.prepare()

Flower_config = flowerConfig()

model = mrcnn.model.MaskRCNN(mode='training',
                             model_dir='./',
                             config=Flower_config)

model.load_weights(filepath='mask_rcnn_coco.h5',
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_set,
            val_dataset=valid_dataset,
            learning_rate=Flower_config.LEARNING_RATE,
            epochs=1,
            layers='heads')

model_path = 'flowerpic_mask_rcnn.h5'
model.keras_model.save_weights(model_path)


# #Image test (Prediction)
# class InferenceConfig(flowerConfig):
#     NAME = "coco_inference"
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     NUM_CLASSES = len(train_set.class_names)
#
#
# inference_config = InferenceConfig()
#
# model = mrcnn.model.MaskRCNN(mode="inference",
#                              config=inference_config,
#                              model_dir=os.getcwd())
#
# model.load_weights(filepath=model_path,
#                    by_name=True)
#
#
# # def get_ax(rows=1, cols=1, size=10):
# #     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
# #
# #     return ax
#
#
# # Test on a random image
# # image_id = random.choice(valid_dataset.image_ids)
# # original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
# # modellib.load_image_gt(valid_dataset, inference_config,image_id, use_mini_mask=False
#
# path_to_new_image = "C:/Users/tracy/PycharmProjects/ImageClassification/Sunflower pic/Validate/aef6c12c05537b0c54135ead8a7700d05fe8ea55.jpg"
# original_image = cv2.imread(path_to_new_image)
# original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
# r = model.detect([original_image], verbose=1)
#
# r = r[0]
#
# mrcnn.visualize.display_instances(image=original_image,
#                                   boxes=r['rois'],
#                                   masks=r['masks'],
#                                   class_ids=r['class_ids'],
#                                   class_names=train_set.class_names,
#                                   scores=r['scores'])
#                                   #ax=get_ax(),
#                                   #title="Predictions")
#
# # plt.savefig("C:/Users/tracy/PycharmProjects/ImageClassification/Sunflower pic/Results/.jpeg")
# # plt.show()
