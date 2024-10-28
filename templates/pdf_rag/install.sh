sudo apt update
sudo apt install -y libgl1-mesa-glx
sudo apt install -y poppler-utils
sudo apt install -y tesseract-ocr
python3 -c 'import nltk; nltk.download("punkt"); nltk.download("averaged_perceptron_tagger")'
