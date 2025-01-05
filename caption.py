import os
import re
import torch
import multiprocessing
import pytesseract
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def get_ocr_text(image_path):
    raw = pytesseract.image_to_string(Image.open(image_path))
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", raw).strip()
    return cleaned

def get_caption(model, processor, tokenizer, device, image_path):
    img = Image.open(image_path).convert("RGB")
    pix = processor(images=img, return_tensors="pt").pixel_values.to(device)
    out_ids = model.generate(pix, max_length=16, num_beams=4)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

def combine_text(ocr_text, caption):
    if len(ocr_text.split()) < 3:
        ocr_text = ""
    if ocr_text and caption:
        combined = f"{ocr_text} {caption}"
    elif ocr_text:
        combined = ocr_text
    elif caption:
        combined = caption
    else:
        combined = "unknown"
    combined = re.sub(r"[^A-Za-z0-9]+", "-", combined)[:50]
    return combined

def main():
    multiprocessing.set_start_method("spawn", force=True)
    folder = "/Users/imac/Desktop/phoname/myimages"
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"folder missing: {folder}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = "nlpconnect/vit-gpt2-image-captioning"

    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            text_ocr = get_ocr_text(path)
            text_cap = get_caption(model, processor, tokenizer, device, path)
            final_name = combine_text(text_ocr, text_cap) + ".jpg"
            new_path = os.path.join(folder, final_name)
            os.rename(path, new_path)
            print(f"renamed {fname} -> {final_name}")

if __name__ == "__main__":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    main()
