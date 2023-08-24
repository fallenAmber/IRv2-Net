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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score, \
    cohen_kappa_score
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
    real_masks = real_masks > threshold
    pred_masks = pred_masks > threshold
    real_masks.astype(np.int32)
    pred_masks.astype(np.int32)

    ## Calculate metrics
    accuracy = accuracy_score(real_masks, pred_masks)
    top_accuracy = top_k_accuracy_score(real_masks, pred_masks, k=10)
    precision = precision_score(real_masks, pred_masks)
    recall = recall_score(real_masks, pred_masks)
    f1 = f1_score(real_masks, pred_masks)

    table_data = [
        ["Accuracy", accuracy],
        ["Top 10 Accuracy", top_accuracy],
        ["Precision", precision],
        ["Recall", recall],
        ["F1", f1]
    ]

    table_header = ["Metric", "Value"]
    print(tabulate(table_data, headers=table_header))

    return accuracy, top_accuracy, precision, recall, f1


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
    pred_masks = pred_masks > threshold
    real_masks.astype(np.int32)
    pred_masks.astype(np.int32)

    ## Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(real_masks, pred_masks)
    pr_auc = auc(recall, precision)

    return precision, recall, pr_auc


def calc_cohen_kappa(real_masks, pred_masks, threshold=0.5):
    real_masks = real_masks.ravel()
    pred_masks = pred_masks.ravel()
    real_masks = real_masks > threshold
    pred_masks = pred_masks > threshold
    real_masks.astype(np.int32)
    pred_masks.astype(np.int32)

    ## Calculate Cohen's kappa score
    kappa = cohen_kappa_score(real_masks, pred_masks)

    return kappa


if __name__ == "__main__":

    # testing 
    # valid_path = "D:/polyp_seg/dataset/new_data_CVC-clinicDB/test"
    valid_path = "D:/polyp_seg/dataset/new_data_Kvasir-SEG/test"

    valid_image_paths = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_mask_paths = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    unet = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/unet_updated_3.h5")
    resu = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/resunet.h5")
    resup = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/resunetplusplus.h5")
    attentionu = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/attentionunet.h5")
    inceptionresu = load_model_weight("D:/polyp_seg/files/models_trained_cvc_clinicDB/inception_resnetv2_unet.h5")

    y0 = get_mask(valid_mask_paths)

    y1 = evaluate_tta(unet, valid_image_paths, valid_mask_paths)
    y2 = evaluate_tta(attentionu, valid_image_paths, valid_mask_paths)
    y3 = evaluate_tta(resu, valid_image_paths, valid_mask_paths)
    y4 = evaluate_tta(resup, valid_image_paths, valid_mask_paths)

    plt.rcParams.update({'font.size': 12})

    y_pred = [y1, y2, y3, y4]
    names = ["UNet", "AttentionUNet", "ResUNet", "ResUNet++"]
    colors = ["g", "r", "y", "b"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    all_metrics = []
    labels = ['Accuracy', 'Top 10 Accuracy', 'Precision', 'Recall', 'F1-score']

    for i in range(len(y_pred)):
        curr_name = names[i]
        c = colors[i]
        print()
        print("Model: " + curr_name)
        accuracy, top_accuracy, precision, recall, f1 = calc_metrics(y0, y_pred[i], threshold=0.5)
        all_metrics.append([accuracy, top_accuracy, precision, recall, f1])

    # Plotting
    width = 0.2
    x = np.arange(len(labels))

    fig, ax = plt.subplots()

    for i, metrics in enumerate(all_metrics):
        ax.bar(x + i * width, metrics, width, label='{}'.format(names[i]))

    ax.set_xticks(x + width * (len(all_metrics) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics by Model')
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="lower right")
    fig.savefig("D:/polyp_seg/irv2_Net/results/visualization.jpg")
    plt.show()
