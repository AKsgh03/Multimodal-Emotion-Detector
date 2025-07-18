# -*- coding: utf-8 -*-
"""Untitled147.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iji2HQ4lxf4DGsGuqoOJZRA3jNZ-tnQf
"""

!pip install datasets transformers torchvision scikit-learn gradio --quiet

import os, random, torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Universal emotion mapping
universal_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_mapping = {
    "admiration": "happy", "amusement": "happy", "approval": "happy", "gratitude": "happy", "joy": "happy", "love": "happy",
    "anger": "angry", "annoyance": "angry", "disapproval": "angry",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "sadness": "sad", "grief": "sad", "remorse": "sad", "disappointment": "sad", "embarrassment": "sad",
    "surprise": "surprise", "realization": "surprise", "excitement": "surprise",
    "neutral": "neutral"
}

# Load dataset
goemotions = load_dataset("go_emotions", split="train")
goemotions_by_emotion = defaultdict(list)

# Map GoEmotions → universal
for entry in goemotions:
    for label_id in entry["labels"]:
        label_name = goemotions.features['labels'].feature.int2str(label_id)
        if label_name in emotion_mapping:
            mapped = emotion_mapping[label_name]
            goemotions_by_emotion[mapped].append(entry["text"])

# Limit to 4000 per class to balance with images
MAX_PER_CLASS = 4000
for emo in universal_emotions:
    random.shuffle(goemotions_by_emotion[emo])
    goemotions_by_emotion[emo] = goemotions_by_emotion[emo][:MAX_PER_CLASS]

# Extract uploaded ZIP
import zipfile
zip_path = "/content/archive (19).zip"  # <-- Change if different
extract_to = "/content/fer_images"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Organize image paths
image_paths_by_emotion = defaultdict(list)
for root, dirs, files in os.walk(extract_to):
    for file in files:
        if file.endswith(".jpg"):
            label = os.path.basename(root).lower()
            if label in universal_emotions:
                image_paths_by_emotion[label].append(os.path.join(root, file))

for emo in universal_emotions:
    random.shuffle(image_paths_by_emotion[emo])
    image_paths_by_emotion[emo] = image_paths_by_emotion[emo][:MAX_PER_CLASS]

paired_data = []
for emo in universal_emotions:
    n = min(len(image_paths_by_emotion[emo]), len(goemotions_by_emotion[emo]))
    texts = goemotions_by_emotion[emo][:n]
    images = image_paths_by_emotion[emo][:n]
    for t, i in zip(texts, images):
        paired_data.append({
            "text": t,
            "image_path": i,
            "label": emo
        })

random.shuffle(paired_data)
label2id = {l: i for i, l in enumerate(universal_emotions)}
id2label = {i: l for l, i in label2id.items()}

from torch.utils.data import Dataset, DataLoader

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MultimodalEmotionDataset(Dataset):
    def __init__(self, data, tokenizer, transform, label2id):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.label2id = label2id

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        image_path = sample["image_path"]
        label = self.label2id[sample["label"]]

        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "image": image,
            "label": torch.tensor(label)
        }

# Create dataset
full_dataset = MultimodalEmotionDataset(paired_data, tokenizer, image_transform, label2id)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

print(f"✅ Final dataset: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision.models import resnet50

class MultimodalFusionClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()

        self.text_proj = nn.Linear(768, 512)
        self.image_proj = nn.Linear(2048, 512)

        self.attn = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, input_ids, attention_mask, image, use_dropout=True):
        txt_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        img_feat = self.resnet(image)

        txt_proj = self.text_proj(txt_feat)
        img_proj = self.image_proj(img_feat)

        if use_dropout and self.training:
            if torch.rand(1).item() < 0.3:
                txt_proj = torch.zeros_like(txt_proj)
            if torch.rand(1).item() < 0.3:
                img_proj = torch.zeros_like(img_proj)

        combined = torch.cat([txt_proj, img_proj], dim=1)
        attn_weights = F.softmax(self.attn(combined), dim=1)
        fused = attn_weights[:, 0].unsqueeze(1) * txt_proj + attn_weights[:, 1].unsqueeze(1) * img_proj

        fused = self.dropout(fused)
        return self.classifier(fused)

from sklearn.metrics import accuracy_score
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, patience=3, model_name="model"):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_preds, train_labels = 0, [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, image=images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds += torch.argmax(outputs, dim=1).cpu().tolist()
            train_labels += labels.cpu().tolist()

        acc = accuracy_score(train_labels, train_preds)
        print(f"✅ Epoch {epoch+1} Train Loss: {train_loss:.4f} | Train Acc: {acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask, image=images, use_dropout=False)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds += torch.argmax(outputs, dim=1).cpu().tolist()
                val_labels += labels.cpu().tolist()

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"📉 Epoch {epoch+1} Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{model_name}_best.pt")
            print(f"💾 Best model saved → {model_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fusion_model = MultimodalFusionClassifier(len(label2id))
optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

train_model(fusion_model, train_loader, val_loader, optimizer, criterion, device, model_name="fusion_model")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, model_name="fusion_model"):
    model.to(device)
    model.load_state_dict(torch.load(f"{model_name}_best.pt"))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, image=images, use_dropout=False)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("✅ Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=universal_emotions))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=universal_emotions, yticklabels=universal_emotions)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

evaluate_model(fusion_model, test_loader, device, model_name="fusion_model")

import gradio as gr

def predict_emotion(text, image):
    fusion_model.eval()
    fusion_model.to(device)

    if text:
        enc = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
    else:
        input_ids = attn_mask = torch.zeros((1, 64), dtype=torch.long).to(device)

    if image:
        image = image.convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(device)
    else:
        image_tensor = torch.zeros((1, 3, 224, 224)).to(device)

    with torch.no_grad():
        output = fusion_model(input_ids=input_ids, attention_mask=attn_mask, image=image_tensor, use_dropout=False)
        pred = torch.argmax(output, dim=1).item()

    return f"🤖 Emotion: {id2label[pred]}"

gr.Interface(
    fn=predict_emotion,
    inputs=[gr.Textbox(label="Text"), gr.Image(type="pil", label="Image")],
    outputs="text",
    title="Multimodal Emotion Detector",
    description="Upload an image and/or enter text to detect emotion using BERT + ResNet50 fusion."
).launch(debug=True)

