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
from m_resunet import ResUnetPlusPlus
from metrics import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
from inception_resnetv2_unet import InceptionResNetV2UNet

from tta import tta_model
from utils import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

THRESHOLD = 0.5


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


def get_mask(y_data):
    total = []
    for y in tqdm(y_data, total=len(y_data)):
        y = read_mask(y)
        total.append(y)
    return np.array(total)


def evaluate_normal(model, x_data, y_data):
    total = []
    for x, y in tqdm(zip(x_data, y_data), total=len(x_data)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(x)[0]  # > THRESHOLD
        y_pred = y_pred.astype(np.float32)
        total.append(y_pred)
    return np.array(total)


def evaluate_tta(model, x_data, y_data):
    total = []
    for x, y in tqdm(zip(x_data, y_data), total=len(x_data)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = tta_model(model, x[0])
        # y_pred = y_pred > THRESHOLD
        y_pred = y_pred.astype(np.float32)
        total.append(y_pred)
    return np.array(total)


def calc_roc(real_masks, pred_masks, threshold=0.5):
    real_masks = real_masks.ravel()

    pred_masks = pred_masks.ravel()
    pred_masks = pred_masks > threshold
    pred_masks.astype(np.int32)

    # ROC AUC Curve
    fpr, tpr, _ = roc_curve(pred_masks, real_masks)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    ## Parameters
    image_size = 256
    batch_size = 32
    lr = 1e-4
    epochs = 5

    ## Validation
    valid_path = "D:/polyp_seg/dataset/new_data_CVC-clinicDB/test"
    # valid_path = "D:/polyp_seg/dataset/new_data_Kvasir-SEG/test"

    valid_image_paths = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_mask_paths = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    # # Kvasir model
    unet = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/unet_kvasir.h5")
    attentionu = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/attentionunet_kvasir.h5")
    resunet = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/resunet_kvasir.h5")
    resunetpp = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/resunetplusplus.h5")
    irv2_net = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/inception_resnetv2_unet.h5")

    # CVC model
    # unet = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/unet_updated_3.h5")
    # attentionu = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/attentionunet.h5")
    # resunet = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/resunet.h5")
    # resunetpp = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/resunetplusplus.h5")
    # irv2_unet = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/inception_resnetv2_unet.h5")

    y0 = get_mask(valid_mask_paths)

    y1 = evaluate_normal(unet, valid_image_paths, valid_mask_paths)
    y2 = evaluate_normal(attentionu, valid_image_paths, valid_mask_paths)
    y3 = evaluate_normal(resunet, valid_image_paths, valid_mask_paths)
    y4 = evaluate_normal(resunetpp, valid_image_paths, valid_mask_paths)
    y5 = evaluate_normal(irv2_net, valid_image_paths, valid_mask_paths)
    y6 = evaluate_tta(irv2_net, valid_image_paths, valid_mask_paths)

    plt.rcParams.update({'font.size': 14})

    y_pred = [y1, y2, y3, y4, y5, y7]
    names = ["UNet", "Attention-Unet", "ResUNet", "ResUNet++", "IRv2-Net", "IRv2-Net+TTA"]
    colors = ["g", "m", "y", "g", "b", "c"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for i in range(len(y_pred)):
        curr_name = names[i]
        c = colors[i]

        fpr, tpr, roc_auc = calc_roc(y0, y_pred[i], threshold=0.5)

        ax.plot(fpr, tpr, c, label=curr_name + ' (area = %0.4f)' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Model Trained on Kvasir and Tested on CVC TestSet')
    ax.legend(loc="lower right")

    fig.savefig("D:/polyp_seg/new_model/ROC-AUC_trained_kVASIR_Test_on_cvc.jpg", dpi=600)
