import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0";
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from metrics import *
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, \
    top_k_accuracy_score
from tta import tta_model
from utils import *
import matplotlib.pyplot as plt
from tabulate import tabulate

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

def calc_metrics(real_masks, pred_masks, threshold=0.5):
    real_masks = real_masks.ravel()
    pred_masks = pred_masks.ravel()
    pred_masks = pred_masks > threshold
    pred_masks.astype(np.int32)

    ## Calculate metrics
    accuracy = accuracy_score(real_masks, pred_masks)
    top_K_accuracy_score = top_k_accuracy_score(real_masks, pred_masks)
    precision = precision_score(real_masks, pred_masks)
    recall = recall_score(real_masks, pred_masks)
    f1 = f1_score(real_masks, pred_masks)

    table_data = [
        ["Accuracy", accuracy],
        ["Top-K Accuracy", top_K_accuracy_score],
        ["Precision", precision],
        ["Recall", recall],
        ["F1", f1]
    ]

    table_header = ["Metric", "Value"]
    print(tabulate(table_data, headers=table_header))

    return accuracy, top_K_accuracy_score, precision, recall, f1


def calc_roc(real_masks, pred_masks, threshold=0.5):
    real_masks = real_masks.ravel()
    pred_masks = pred_masks.ravel()
    pred_masks = pred_masks > threshold
    pred_masks.astype(np.int32)

    ## ROC AUC Curve
    fpr, tpr, _ = roc_curve(pred_masks, real_masks)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def calc_precision_recall(real_masks, pred_masks, threshold=0.5):
    real_masks = real_masks.ravel()
    pred_masks = pred_masks.ravel()
    real_masks = real_masks > threshold
    real_masks.astype(np.int32)
    pred_masks = pred_masks > threshold
    pred_masks.astype(np.int32)

    ## Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(real_masks, pred_masks)
    pr_auc = auc(recall, precision)

    return precision, recall, pr_auc


def calc_cohen_kappa(real_masks, pred_masks, threshold=0.5):
    real_masks = real_masks.ravel()
    pred_masks = pred_masks.ravel()
    real_masks = real_masks > threshold
    real_masks.astype(np.int32)
    pred_masks = pred_masks > threshold
    pred_masks.astype(np.int32)

    ## Calculate Cohen's kappa score
    kappa = cohen_kappa_score(real_masks, pred_masks)

    return kappa


if __name__ == "__main__":

    ## Testing
    valid_path = "D:/polyp_seg/dataset/new_data_CVC-clinicDB/test"
    # valid_path = "D:/polyp_seg/dataset/new_data_Kvasir-SEG/test"

    valid_image_paths = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_mask_paths = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    # # Kvasir model
    unet = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/unet_kvasir.h5")
    attentionu = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/attentionunet_kvasir.h5")
    resunet = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/resunet_kvasir.h5")
    resunetpp = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/resunetplusplus_kvasir.h5")
    irv2_unet = load_model_weight("D:/polyp_seg/files/models_trained_kvasir_seg/inception_resnetv2_unet_kvasir.h5")

    # CVC model
    # unet = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/unet_updated_3.h5")
    # attentionu = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/attentionunet.h5")
    # resunet = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/resunet.h5")
    # resunetpp = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/resunetplusplus.h5")
    # irv2_unet = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/inception_resnetv2_unet.h5")

    y0 = get_mask(valid_mask_paths)

    # y1 = evaluate_normal(unet, valid_image_paths, valid_mask_paths)
    # y2 = evaluate_normal(attentionu, valid_image_paths, valid_mask_paths)
    # y3 = evaluate_normal(resunet, valid_image_paths, valid_mask_paths)
    # y4 = evaluate_normal(resunetpp, valid_image_paths, valid_mask_paths)
    # y5 = evaluate_normal(irv2_unet, valid_image_paths, valid_mask_paths)
    # y6 = evaluate_tta(irv2_unet, valid_image_paths, valid_mask_paths)

    y1 = evaluate_normal(unet, valid_image_paths, valid_mask_paths)
    y2 = evaluate_normal(attentionu, valid_image_paths, valid_mask_paths)
    y3 = evaluate_normal(resunet, valid_image_paths, valid_mask_paths)
    y4 = evaluate_normal(resunetpp, valid_image_paths, valid_mask_paths)
    y5 = evaluate_normal(irv2_unet, valid_image_paths, valid_mask_paths)            ##====
    y7 = evaluate_tta(irv2_unet, valid_image_paths, valid_mask_paths)

    plt.rcParams.update({'font.size': 15})

    y_pred = [y1, y2, y3, y4, y5, y7]
    names = ["UNet", "Attention-Unet", "ResUNet", "ResUNet++", "IRv2-Net", "IRv2-Net+TTA"]
    colors = ["g", "m", "y", "g", "b", "c"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for i in range(len(y_pred)):
        curr_name = names[i]
        c = colors[i]

        precision, recall, pr_auc = calc_precision_recall(y0, y_pred[i], threshold=0.5)

        ax.plot(precision, recall, label=curr_name + ' (area = %0.4f)' % pr_auc)
        ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Model Trained on Kvasir and Tested on CVC TestSet')
    ax.legend(loc="lower left")

    fig.savefig("D:/polyp_seg/new_model/AUC-PR_irv2-net.jpg", dpi=600)
