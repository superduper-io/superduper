# server.py
import asyncio
import os
import re
import sys
import time
import datetime
import difflib
import json
import redis
import threading

from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from superduper import superduper, logging, CFG
CFG.log_colorize = False

from superduper.components.component import Component


class Question(BaseModel):
    filename: str
    question: str

###############################################################################
# 0. READ COMMAND-LINE ARG FOR WATCH PATH
###############################################################################


if len(sys.argv) > 1:
    WATCH_PATH = sys.argv[1]
    if WATCH_PATH == '.':
        WATCH_PATH = os.getcwd()
else:
    WATCH_PATH = os.getcwd()

PROJECT_NAME = WATCH_PATH.split('/')[-1]

IGNORE = '^files|^artifact_store|^.superduper_pair_programmer'

app = FastAPI()

###############################################################################
# FILE WATCHER FUNCTIONALITY
###############################################################################
def diff(old_content: str, new_content: str) -> str:
    lines1 = old_content.splitlines(keepends=True)
    lines2 = new_content.splitlines(keepends=True)
    d = list(
        difflib.unified_diff(
            lines1,
            lines2,
            fromfile='old_file',
            tofile='new_file',
            lineterm='',
            n=100000
        )
    )
    return '\n'.join(d)


class Handler(FileSystemEventHandler):
    def __init__(self, db):
        super().__init__()
        self.db = db

    def on_any_event(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith('.py'):
            return

        with open(event.src_path, 'r') as file:
            content = file.read()

        if event.event_type == 'created':
            self.db['files_' + PROJECT_NAME].insert([{
                'filename': event.src_path,
                'content': content,
                'last_modified': str(datetime.datetime.now()),
                'diff': None
            }])
            logging.info(f"Received created event - {event.src_path}.")
            return

        filename = event.src_path.split('/')[-1]
        previous = self.db['files_' + PROJECT_NAME].get(filename=filename)

        if event.event_type == 'modified':
            logging.info(f"Received modified event - {event.src_path}.")
            repl = {
                'filename': filename,
                'content': content,
                'last_modified': str(datetime.datetime.now()),
                'diff': diff(previous['content'], content) if previous else None,
            }
            self.db['files_' + PROJECT_NAME].replace({'filename': filename}, repl)

        elif event.event_type == 'deleted':
            self.db['files_' + PROJECT_NAME].delete(filename=event.src_path)
            logging.info(f"Received deleted event - {event.src_path}.")


class ApplyApplicationHandler(FileSystemEventHandler):
    def __init__(self, db):
        super().__init__()
        self.db = db

    def on_any_event(self, event):

        logging.info(f"Received event - {event.event_type} - {event.src_path}")
        if not event.src_path.endswith('component.yaml'):
            return

        if event.event_type == 'created':
            return

        logging.info('Applying updated application')
        path = '/'.join(event.src_path.split('/')[:-1])
        application = Component.read(path)
        self.db.apply(application, force=True)
        logging.info('Applying updated application... DONE')


class Watcher:
    def __init__(self, directory_to_watch, db, handler=None):
        self.observer = Observer()
        self.directory_to_watch = directory_to_watch
        self.db = db
        self.handler = handler

    def run(self):
        event_handler = self.handler or Handler(db=self.db)
        self.observer.schedule(event_handler, self.directory_to_watch, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
        except Exception as e:
            logging.error("Error in watcher: " + str(e))
            self.observer.stop()
        self.observer.join()

###############################################################################
# WEBSOCKET / FASTAPI
###############################################################################
clients = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logging.info(f"Received message: {data}")
    except WebSocketDisconnect:
        clients.remove(websocket)
        logging.info("Client disconnected")


@app.post("/ask_superduper")
def ask_superduper(question: Question):
    if question.filename.startswith('/'):
        filename = question.filename[len(WATCH_PATH):]
    else:
        filename = question.filename
    logging.info(f"Received question: {question} for file: {filename}")
    db = app.state.db
    m = db.load('AskSuperduper', 'ask_' + PROJECT_NAME)
    answer = m.predict(question=question.question, filename=filename)
    return {'answer': answer}


async def distribute_message(message: str):
    """Send messages to all connected WebSocket clients."""
    disconnected_clients = []
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            # If we fail to send, it means the client is disconnected or invalid.
            disconnected_clients.append(client)

    # Clean up clients that have disconnected
    for dc in disconnected_clients:
        clients.remove(dc)


def subscribe_to_key_events(redis_client, prefixes):
    pubsub = redis_client.pubsub()
    for prefix in prefixes:
        pubsub.psubscribe(f"__key*__:{prefix}*")
    return pubsub


async def redis_listener(redis_client, prefixes):
    pubsub = subscribe_to_key_events(redis_client, prefixes)
    try:
        while True:
            message = pubsub.get_message()
            if message and message['type'] == 'pmessage':
                channel = message['channel'].decode("utf-8")
                event = message['data'].decode("utf-8")
                parts = channel.split(":", 2)
                if len(parts) >= 3:
                    redis_key = parts[1] + ":" + parts[2]
                    if event == "json.set":
                        value_object = redis_client.json().get(redis_key)
                        if value_object is not None:
                            json_string = json.dumps(value_object)
                            await distribute_message(json_string)
            await asyncio.sleep(0.1)
    finally:
        pubsub.close()


def update_application_if_changed(db):
    logging.info("Starting application update thread.")
    os.makedirs(WATCH_PATH + '/.superduper_pair_programmer', exist_ok=True)
    watcher = Watcher(
        WATCH_PATH + '/.superduper_pair_programmer',
        db=db,
        handler=ApplyApplicationHandler(db=db),
    )
    watcher.run()


###############################################################################
# BACKGROUND INITIALIZATION
###############################################################################
def background_init(db, path="."):
    """
    1. Insert existing .py files into DB (if not already present).
    2. Start the Watcher (blocking).
    """
    now = datetime.datetime.now()

    if not path.endswith('/'):
        path += '/'

    files = [
        os.path.join(dp, f)
        for dp, _, fn in os.walk(path)
        for f in fn if f.endswith('.py')
    ]
    files = [f.replace("./", "") for f in files]
    files = [re.sub('^' + path, '', f) for f in files]
    files = [f for f in files if not re.match(IGNORE, f)]

    dont_exist = []
    for file in files:
        if db['files_' + PROJECT_NAME].get(filename=file):
            continue
        dont_exist.append(file)

    if dont_exist:
        db['files_' + PROJECT_NAME].insert([
            {
                'filename': file,
                'content': open(path + file, 'r').read(),
                'last_modified': str(now),
                'diff': None
            }
            for file in dont_exist
        ])
    logging.info("Finished initial scanning of .py files.")

    # Now start the actual file watcher
    watcher = Watcher(path, db=db)
    watcher.run()

###############################################################################
# STARTUP EVENT
###############################################################################
@app.on_event("startup")
async def startup_event():
    # 1) Connect to Redis
    redis_client = redis.Redis.from_url('redis://localhost:6379/0')
    redis_client.config_set('notify-keyspace-events', 'KEA')
    prefixes = [f'comments_{PROJECT_NAME}:']
    asyncio.create_task(redis_listener(redis_client, prefixes))

    # 2) Create your superduper DB
    db = superduper('redis://localhost:6379/0')

    app.state.db = db

    if PROJECT_NAME not in db.show('Application'):
        from superduper import Template
        t = Template.read(
            os.path.expanduser('~/superduper-io/superduper/templates/copilot')
        )
        application = t(project_name=PROJECT_NAME)
        db.apply(application, force=True)

    os.makedirs('.superduper_pair_programmer', exist_ok=True)
    if not os.path.exists('.superduper_pair_programmer/component.yaml'):
        application = db.load('Application', PROJECT_NAME)
        application.export('.superduper_pair_programmer', format='yaml')

    # 3) Run scanning + watcher in a background thread
    init_thread = threading.Thread(
        target=background_init, 
        args=(db, WATCH_PATH), 
        daemon=True
    )
    init_thread.start()

    update_thread = threading.Thread(
        target=update_application_if_changed, 
        args=(db,), 
        daemon=True
    )
    update_thread.start()

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    # Example usage:
    #    python server.py /path/to/watch
    #
    # If no path is specified, it defaults to "." above.
    uvicorn.run(app, host="127.0.0.1", port=8000)
