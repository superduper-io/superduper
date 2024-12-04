## PDF RAG template

**Clone the code**

```bash
git clone --branch main --depth 1 git@github.com:superduper-io/superduper.git
```

**Build the docker**

```bash
docker build . -f templates/pdf_rag/Dockerfile -t pdf-rag
```

**Start the docker container with the ports opened and data mounted**

You need to mount a local directory to `/app/data` in the container.

***The PDF directory should be a sub-directory of this directory!***

```bash
docker run -it -p 8501:8501 -v ./data/:/app/data pdf-rag bash
```

**Set your OpenAI or Ollama credentials**

(Run this in the container)

```bash
export OPENAI_API_KEY=ollama  # or sk-<secret> for openai
export OPENAI_BASE_URL=...    # URL of ollama server if applicable
```

**Set your MongoDB connection**

```bash
export SUPERDUPER_DATA_BACKEND=mongodb://host.docker.internal:27017/test_db
export SUPERDUPER_ARTIFACT_STORE=filesystem://./data
```

**Prepare the app with your choice of models and data**

```bash
bash templates/pdf_rag/start.sh bodybuilder <embedding_model> <llm_model>
```

For example, on OpenAI:

```bash
bash templates/pdf_rag/start.sh bodybuilder text-embedding-ada-002 gpt-3.5-turbo
```

For example, on Ollama:

```bash
bash templates/pdf_rag/start.sh bodybuilder nomic-embed-text:latest llama3.1:70b
```

**Run the app's frontend**

```bash
python3 -m streamlit run templates/pdf-rag/streamlit.py
```

### Notes

- If you exit the docker container, you will need to reset your environment variables (`export $...`)
- If you are using Ollama, you need to use models which are installed on the server
