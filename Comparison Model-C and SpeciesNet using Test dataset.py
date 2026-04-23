# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:47:57 2026

@author: k4927
"""

# Comparison Model-C and SpeciesNet using Test dataset

!pip -q install -U kagglehub speciesnet

import os

os.environ["KAGGLEHUB_CACHE"] = "/content/kagglehub_cache"
os.makedirs("/content/kagglehub_cache", exist_ok=True)
os.makedirs("/content/output", exist_ok=True)

print("KAGGLEHUB_CACHE =", os.environ["KAGGLEHUB_CACHE"])

import kagglehub

MODEL_PATH = kagglehub.model_download(
    "google/speciesnet/pyTorch/v4.0.2a/1",
    output_dir="/content/speciesnet_models",
    force_download=True
)

print("MODEL_PATH =", MODEL_PATH)

import shutil
shutil.copytree("/content/drive/MyDrive/WFdata_Boar,Deer(2208_2212)/Test_120","/content/Test120")

image_folder = "/content/Test120"
output_json = "/content/output/speciesnet_output.json"

import os
print("image exists:", os.path.exists(image_folder))
print("output dir exists:", os.path.exists("/content/output"))

# Run SpeciesNet
!python -m speciesnet.scripts.run_model \
  --model "$MODEL_PATH" \
  --folders "$image_folder" \
  --predictions_json "$output_json" \
  --country KOR

# SpeciesNet prediction resutls
import json
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

json_path = "/content/output/speciesnet_output.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

target_labels = ["badger", "wildboar", "raccoon", "roedeer", "waterdeer"]

def get_true_label(filepath):
    name = os.path.basename(filepath).lower()
    if "__bad__" in name:
        return "badger"
    elif "__wild__" in name:
        return "wildboar"
    elif "__raccoon__" in name:
        return "raccoon"
    elif "__roe__" in name:
        return "roedeer"
    elif "__water__" in name:
        return "waterdeer"
    else:
        return "other"

def get_pred_label_from_top1(item):
    top1 = item["classifications"]["classes"][0].lower()

    if "badger" in top1:
        return "badger"
    elif "wild boar" in top1:
        return "wildboar"
    elif ("raccoon dog" in top1) or ("northern raccoon" in top1):
        return "raccoon"
    elif "water deer" in top1:
        return "waterdeer"
    elif "roe deer" in top1:
        return "roedeer"
    else:
        return "other"

rows = []
for item in data["predictions"]:
    true_label = get_true_label(item["filepath"])
    pred_label = get_pred_label_from_top1(item)

    if true_label in target_labels:
        rows.append({
            "filepath": item["filepath"],
            "true": true_label,
            "pred": pred_label
        })

df = pd.DataFrame(rows)
print(df.head())
print("N =", len(df))

# Draw SpeciesNet Confusion matrix 
import json
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

json_path = "/content/output/speciesnet_output.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 원하는 순서
labels_6 = ["badger", "raccoondog", "boar", "roedeer", "waterdeer", "background"]

def get_true_label(filepath):
    name = os.path.basename(filepath).lower()
    if "__bad__" in name:
        return "badger"
    elif "__wild__" in name:
        return "boar"
    elif "__raccoon__" in name:
        return "raccoondog"
    elif "__roe__" in name:
        return "roedeer"
    elif "__water__" in name:
        return "waterdeer"
    elif "__background__" in name:
        return "background"
    else:
        return None

def get_pred_label_top1_6class(item):
    top1 = item["classifications"]["classes"][0].lower()

    if "badger" in top1:
        return "badger"
    elif "wild boar" in top1:
        return "boar"
    elif ("raccoon dog" in top1) or ("northern raccoon" in top1):
        return "raccoondog"
    elif "roe deer" in top1:
        return "roedeer"
    elif "water deer" in top1:
        return "waterdeer"
    elif "blank" in top1:
        return "background"
    else:
        return "background"

rows = []
for item in data["predictions"]:
    true_label = get_true_label(item["filepath"])
    if true_label is None:
        continue
    pred_label = get_pred_label_top1_6class(item)
    rows.append({
        "filepath": item["filepath"],
        "true": true_label,
        "pred": pred_label
    })

df = pd.DataFrame(rows)

cm = confusion_matrix(df["true"], df["pred"], labels=labels_6)
cm_df = pd.DataFrame(cm, index=labels_6, columns=labels_6)
cm_df_swapped = cm_df.T
cm_ratio = cm_df.div(cm_df.sum(axis=1), axis=0)
cm_ratio_swapped = cm_ratio.T.round(3)

print("Count confusion matrix (rows=Predicted, cols=True)")
display(cm_df_swapped)

print("Proportion confusion matrix (rows=Predicted, cols=True)")
display(cm_ratio_swapped)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_ratio_swapped.values, cmap="Blues", vmin=0, vmax=1)

ax.set_xticks(range(len(labels_6)))
ax.set_yticks(range(len(labels_6)))
ax.set_xticklabels(labels_6, rotation=45, ha="right")
ax.set_yticklabels(labels_6)

for i in range(cm_ratio_swapped.shape[0]):
    for j in range(cm_ratio_swapped.shape[1]):
        val = cm_ratio_swapped.iloc[i, j]
        color = "white" if val >= 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
# ax.set_title("SpeciesNet Confusion Matrix (%)")
plt.colorbar(im)
plt.tight_layout()
plt.show()


# Model-C grouped K-fold Cross validation 