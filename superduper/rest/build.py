import hashlib
import json
import os
import shutil
import sys
import tempfile
import typing as t
import zipfile
from contextlib import contextmanager

import magic
from fastapi import File, Response

from superduper import logging
from superduper.backends.base.query import Query
from superduper.base.document import Document
from superduper.components.component import Component
from superduper.components.template import Template
from superduper.rest.base import SuperDuperApp

from .utils import rewrite_artifacts


class Tee:
    """A file-like object that writes to multiple outputs."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()  # Ensure each write is flushed immediately

    def flush(self):
        for f in self.files:
            f.flush()


@contextmanager
def redirect_stdout_to_file(file_path: str):
    """Context manager to redirect stdout to a specified file temporarily."""
    original_stdout = sys.stdout
    try:
        with open(file_path, 'w', buffering=1) as f:
            sys.stdout = Tee(original_stdout, f)
            yield
    finally:
        sys.stdout = original_stdout


def build_rest_app(app: SuperDuperApp):
    """
    Add the key endpoints to the FastAPI app.

    :param app: SuperDuperApp
    """

    @app.add('/db/artifact_store/put', method='put')
    def db_artifact_store_put_bytes(raw: bytes = File(...)):
        file_id = str(hashlib.sha1(raw).hexdigest())
        app.db.artifact_store.put_bytes(serialized=raw, file_id=file_id)
        return {'file_id': file_id}

    @app.add('/db/artifact_store/get', method='get')
    def db_artifact_store_get_bytes(file_id: str):
        bytes = app.db.artifact_store.get_bytes(file_id=file_id)
        media_type = magic.from_buffer(bytes, mime=True)
        return Response(content=bytes, media_type=media_type)

    @app.add("/db/upload", method="put")
    def db_upload(raw: bytes = File(...)):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "component.zip")
            with open(path, "wb") as f:
                f.write(raw)

            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            path = os.path.join(temp_dir, "component.json")
            try:
                with open(path, "r") as f:
                    component = json.load(f)
            except FileNotFoundError as f:
                raise Exception("No component.json file found in the zip file")

            blobs = os.path.join(temp_dir, "blobs")
            blob_objects = []
            if os.path.exists(blobs):
                blob_objects = os.listdir(blobs)
                for blob in blob_objects:
                    blob_path = os.path.join(blobs, blob)
                    with open(blob_path, "rb") as f:
                        content = f.read()
                    app.db.artifact_store.put_bytes(content, blob)

            files = os.path.join(temp_dir, "files")
            if os.path.exists(files):
                file_objects = os.listdir(files)
                for file in file_objects:
                    file_path = os.path.join(files, file)
                    app.db.artifact_store.put_file(file_path, file)

        # TODO add file objects
        # Component to be rendered in the front end
        # Blob objects to be displayed on the upload
        return {"component": component, "artifacts": blob_objects}

    def _print_to_screen():
        for i in range(100):
            print(f'Testing {i}')
            time.sleep(0.1)
        print('[DONE]')

    @app.add('/test/log', method='post')
    def test_log():
        log_file = "/tmp/test.log"
        with redirect_stdout_to_file(log_file):
            _print_to_screen()
        os.remove(log_file)
        return {'status': 'ok'}

    def _process_db_apply(info):
        if '_variables' in info:
            assert {'_variables', 'identifier'}.issubset(info.keys())
            variables = info.pop('_variables')
            for k in variables:
                assert '<' not in variables[k]
                assert '>' not in variables[k]
                assert ' ' not in variables[k]

            identifier = info.pop('identifier')
            template_name = info.pop('_template_name', None)

            component = Component.from_template(
                identifier=identifier,
                template_body=info,
                template_name=template_name,
                db=app.db,
                **variables,
            )
            app.db.apply(component, force=True)
            return {'status': 'ok'}
        component = Document.decode(info, db=app.db).unpack()
        # TODO this shouldn't be necessary to do twice
        component.unpack()
        app.db.apply(component, force=True)
        return {'status': 'ok'}

    @app.add('/db/apply', method='post')
    def db_apply(info: t.Dict, id: str | None = None):
        if id:
            log_file = f"/tmp/{id}.log"
            with redirect_stdout_to_file(log_file):
                try:
                    out = _process_db_apply(info)
                finally:
                    os.remove(log_file)
        else:
            out = _process_db_apply(info)
        return out

    import subprocess
    import time

    from fastapi.responses import StreamingResponse

    def tail_f(filename):
        process = subprocess.Popen(
            ["tail", "-n", "1000", "-f", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            for line in process.stdout:
                yield line
                if '[DONE]' in line:
                    process.terminate()
                    return
        except KeyboardInterrupt:
            process.terminate()

    @app.add('/stdout', method='get')
    async def stream(id: str):
        return StreamingResponse(
            tail_f(f'/tmp/{id}.log'), media_type="text/event-stream"
        )

    @app.add('/db/show', method='get')
    def db_show(
        type_id: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
        application: t.Optional[str] = None,
    ):
        if type_id == 'template' and identifier is None:
            out = app.db.show('template')
            from superduper import templates

            out += [x for x in templates.ls() if x not in out]
            return out
        if application is not None:
            r = app.db.metadata.get_component('application', application)
            return r['namespace']
        else:
            return app.db.show(
                type_id=type_id,
                identifier=identifier,
                version=version,
            )

    @app.add('/db/remove', method='post')
    def db_remove(type_id: str, identifier: str):
        app.db.remove(type_id=type_id, identifier=identifier, force=True)
        return {'status': 'ok'}

    @app.add('/db/show_template', method='get')
    def db_show_template(identifier: str, type_id: str = 'template'):
        template: Template = app.db.load(type_id=type_id, identifier=identifier)
        return template.form_template

    @app.add('/db/metadata/show_jobs', method='get')
    def db_metadata_show_jobs(type_id: str, identifier: t.Optional[str] = None):
        return [
            r['job_id']
            for r in app.db.metadata.show_jobs(type_id=type_id, identifier=identifier)
            if 'job_id' in r
        ]

    @app.add('/db/execute', method='post')
    def db_execute(
        query: t.Dict,
    ):
        if query['query'].startswith('db.show'):
            output = eval(f'app.{query["query"]}')
            logging.info('db.show results:')
            logging.info(output)
            return [{'_base': output}], []

        if '_path' not in query:
            plugin = app.db.databackend.type.__module__.split('.')[0]
            query['_path'] = f'{plugin}.query.parse_query'

        q = Document.decode(query, db=app.db).unpack()

        logging.info('processing this query:')
        logging.info(q)

        result = q.execute()

        if q.type in {'insert', 'delete', 'update'}:
            return {'_base': [str(x) for x in result[0]]}, []

        logging.warn(str(q))

        if isinstance(result, Document):
            result = [result]

        result = [rewrite_artifacts(r, db=app.db) for r in result]
        result = [r.encode() for r in result]
        blobs_keys = [list(r.pop_blobs().keys()) for r in result]
        result = list(zip(result, blobs_keys))

        if isinstance(q, Query):
            for i, r in enumerate(result):
                r = list(r)
                if q.primary_id in r[0]:
                    r[0][q.primary_id] = str(r[0][q.primary_id])
                result[i] = tuple(r)
            if 'score' in result[0][0]:
                result = sorted(result, key=lambda x: -x[0]['score'])
        return result


def build_frontend(app: SuperDuperApp, host: str = 'localhost', port: int = 8000):
    """Add the frontend to the FastAPI app.

    :param app: app instance SuperDuperApp
    :param host: host address
    :param port: port number
    """
    import os

    from fastapi import HTTPException, Request
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    ROOT = os.path.dirname(os.path.abspath(__file__))

    try:
        shutil.rmtree(f"{ROOT}/superdupertmp")
    except FileNotFoundError:
        pass

    shutil.copytree(f"{ROOT}/out", f"{ROOT}/superdupertmp")

    DIRECTORY = f"{ROOT}/superdupertmp"

    if host != 'localhost' or port != 8000:
        for root, _, files in os.walk(DIRECTORY):
            for file in files:
                if file.endswith('.js'):
                    with open(os.path.join(root, file), "r") as f:
                        content = f.read()
                    content = content.replace("localhost:8000", f"{ host }:{ port }")
                    with open(os.path.join(root, file), "w") as f:
                        f.write(content)

    app.app.mount("/static", StaticFiles(directory=DIRECTORY), name="static")

    @app.app.get("/{path:path}")
    async def serve_file(request: Request, path: str):
        """Serve files from the default 'out' directory.

        :param request: Request
        :param path: path to file
        """
        # Special case: if path is 'webui', serve the 'index.html'
        # from the 'out' directory
        if path == "webui":
            webui_index = os.path.join(DIRECTORY, "index.html")
            if os.path.exists(webui_index):
                return FileResponse(webui_index)
            else:
                raise HTTPException(
                    status_code=404, detail="index.html not found for /webui"
                )

        # Normal case: serve files from the 'out' directory
        requested_path = os.path.join(DIRECTORY, path.lstrip("/"))

        # If the path is a directory, attempt to serve index.html
        if os.path.isdir(requested_path):
            index_file = os.path.join(requested_path, "index.html")
            if os.path.exists(index_file):
                return FileResponse(index_file)

        if os.path.exists(requested_path):
            return FileResponse(requested_path)

        # Try appending .html to the requested path
        path_with_html = f"{requested_path}.html"
        if os.path.exists(path_with_html):
            return FileResponse(path_with_html)

        # If file not found, raise a 404 error
        raise HTTPException(status_code=404, detail="File not found")
