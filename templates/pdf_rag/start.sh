mkdir -p data/artifacts
superduper bootstrap ./templates/pdf_rag
python3 templates/pdf_rag/add_data.py data/$1
superduper apply pdf-rag --variables '{"table_name": "'$1'", "embedding_model": "'$2'", "llm_model": "'$3'"}'