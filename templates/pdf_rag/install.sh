apt update
apt install -y libgl1-mesa-glx
apt install -y poppler-utils
apt install -y tesseract-ocr
python3 -c 'import nltk; nltk.download("punkt"); nltk.download("averaged_perceptron_tagger")'
