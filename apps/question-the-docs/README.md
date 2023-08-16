# Question the Docs :book:

This app introduces the FARMS stack - FastAPI, React, MongoDB and SuperDuperDB. Full details on the FARM stack are available [here](https://www.mongodb.com/developer/languages/python/farm-stack-fastapi-react-mongodb/).

## Frontend :art:

The frontend has been developed with Node.js version 18.17.1. The packages can be installed with `npm install --prefix frontend/` and the app run with `npm run dev --prefix frontend`.

## Backend :computer:

The backend has been developed with CPython 3.8. To begin, you will need to create a GitHub PAT token and set this as an environment variable (`GITHUB_TOKEN`) in your local environment. This token is required for interacting with the GitHub API. See `backend/ai/utils/github.py` for details.

Next, you will need to setup an account with MongoDB Atlas and configure a cluster for access with the app. You should set the URI for this cluster as an environment variable (`mongo_uri`). If all goes well, you should end with something like `mongo_uri="mongodb+srv://<USER>:<PASSWORD>@<CLUSTER>.qwekqo3.mongodb.net/<DB>?retryWrites=true&w=majority"`. Please contact Timo if there are any issues at this stage.

Finally, you will also need to create an OpenAI account, get a token and set this as an environment variable (`OPENAI_API_KEY`).

After you have set these environment variables, to run the backend, install the Python environment with the command `python -m pip -r backend/requirements.in` and start the webserver (eg `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`).

Good luck! :rocket: