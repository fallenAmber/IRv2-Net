import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0";
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.metrics import MeanIoU

import tensorflow.keras.backend as K
from inception_resnetv2_unet import InceptionResNetV2UNet
from metrics import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
from tta import tta_model


def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def get_dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def get_mean_iou(y_true, y_pred):

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_true = y_true.astype(np.int32)
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(y_true, y_pred)
    r = m.result().numpy()
    m.reset_states()
    return r


def get_recall(y_true, y_pred):

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    m = tf.keras.metrics.Recall()
    m.update_state(y_true, y_pred)
    r = m.result().numpy()
    m.reset_states()
    return r


def get_precision(y_true, y_pred):

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    m = tf.keras.metrics.Precision()
    m.update_state(y_true, y_pred)
    r = m.result().numpy()
    m.reset_states()
    return r


def confusion(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return prec, recall


def get_metrics(y_true, y_pred):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    dice_coef_val = get_dice_coef(y_true, y_pred)
    mean_iou_val = get_mean_iou(y_true, y_pred)

    y_true = y_true.astype(np.int32)

    recall_value = get_recall(y_true, y_pred)
    precision_value = get_precision(y_true, y_pred)

    return [dice_coef_val, mean_iou_val, recall_value, precision_value]


def evaluate_normal(model, x_data, y_data):
    total = []
    for x, y in tqdm(zip(x_data, y_data), total=len(x_data)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(x)[0] > 0.5
        y_pred = y_pred.astype(np.float32)

        value = get_metrics(y, y_pred)
        total.append(value)

    mean_value = np.mean(total, axis=0)
    print(mean_value)


def evaluate_tta(model, x_data, y_data):
    total = []
    for x, y in tqdm(zip(x_data, y_data), total=len(x_data)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = tta_model(model, x[0])
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.float32)

        value = get_metrics(y, y_pred)
        total.append(value)

    mean_value = np.mean(total, axis=0)
    print(mean_value)


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    model_path = "D:/polyp_seg/files/models_trained_Kvasir_SEG/inception_resnetv2_unet.h5"
    
    ## Parameters
    image_size = 256
    batch_size = 32
    lr = 1e-4
    epochs = 50

    ## Testing 
    # valid_path = "D:/polyp_seg/dataset/new_data_CVC-clinicDB/test"
    valid_path = "D:/polyp_seg/dataset/new_data_Kvasir-SEG/test"

    valid_image_paths = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_mask_paths = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    with CustomObjectScope({
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'bce_dice_loss': bce_dice_loss,
        'focal_loss': focal_loss,
        'tversky_loss': tversky_loss,
        'focal_tversky': focal_tversky
        }):

        model = tf.keras.models.load_model(model_path)

    evaluate_normal(model, valid_image_paths, valid_mask_paths)
    print("Base Model")

    evaluate_tta(model, valid_image_paths, valid_mask_paths)
    print("Base+TTA")
