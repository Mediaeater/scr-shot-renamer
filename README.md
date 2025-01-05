# image-rename: local ocr + ai captioner

this script merges offline ocr + huggingface image captioning to rename images with more descriptive filenames.

## features
- extracts text from images with [pytesseract](https://github.com/madmaze/pytesseract).
- uses the [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) model to generate short captions.
- merges both results into a final filename (e.g. `MeetingAgendaTeam-Photo.jpg`).

## requirements
- **python >= 3.10**
- **tesseract** installed locally  
  - on mac: `brew install tesseract`
- **pytorch** with mps support (apple silicon), e.g.:  
  ```bash
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu


```bash
 pip install transformers pillow pytesseract


##usage
edit caption.py to set folder to your image directory. (create if needed)
confirm tesseract is installed: tesseract --version.
run the script:

  ```bash
python caption.py

- ** each image will be renamed in-place with a combination of ocr text and ai-generated caption.
 
