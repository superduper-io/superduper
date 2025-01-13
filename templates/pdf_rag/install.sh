sudo apt update
sudo apt install -y libgl1-mesa-glx
sudo apt install -y poppler-utils
sudo apt install -y tesseract-ocr
python -c 'import nltk; nltk.download("punkt"); nltk.download("averaged_perceptron_tagger"); nltk.download("punkt_tab"); nltk.download("averaged_perceptron_tagger_eng");'
