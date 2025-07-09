# Multimodal-Emotion-Detector
A full-scale deep learning project that detects human emotions by analyzing both textual data (speech transcripts, messages) and facial expressions from images.
This multimodal approach boosts accuracy by combining semantic and visual cues.

Key Features:
Text Modality: Utilizes a fine-tuned BERT model on the GoEmotions dataset to classify emotions from sentences.

Image Modality: Uses ResNet-50 CNN trained on FER-style datasets to detect facial emotions from images.

Multimodal Fusion: Combines predictions from both modalities using decision-level fusion for robust and consistent emotion classification.

Evaluation & Visualization: Includes precision, recall, F1-score metrics, and confusion matrices for individual and combined models.

Deployment-Ready Pipeline: Clean code structure, inference scripts, and ready for deployment with Streamlit or Flask.

Tech Stack:
Python, PyTorch, Transformers (Hugging Face), OpenCV

Pretrained Models: BERT, ResNet50

Datasets: GoEmotions (text), FER+ / custom FER dataset (images)

