# Encoder-only model: Multilingual BERT (sentiment classification)

- Model: https://huggingface.co/TahiraAbbas/bert-multilingual-imdb-sentiment
- Base model: `bert-base-multilingual-cased` (104 languages)
- Dataset: IMDB movie reviews (positive / negative)
- Live app: PASTE_YOUR_BERT_STREAMLIT_LINK_HERE

Files: `app.py` (Streamlit app), `requirements.txt`, `sample_reviews.csv` (30 example reviews), `Encoder_BERT_Colab.ipynb` (fine-tuning code).

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`
