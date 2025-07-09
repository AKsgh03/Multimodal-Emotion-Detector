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
