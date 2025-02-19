import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import typing as t
import zipfile
from contextlib import contextmanager

from superduper.base.exceptions import IncorrectSecretException
from superduper.misc.files import check_secrets, load_secrets
from superduper.misc.importing import import_object
from superduper.misc.plugins import load_plugin

try:
    import magic
except ImportError:
    magic = None
from fastapi import BackgroundTasks, File, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_409_CONFLICT,
    HTTP_410_GONE,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base.document import Document
from superduper.components.template import Template
from superduper.rest.base import DatalayerDependency, SuperDuperApp

from .utils import rewrite_artifacts

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


PENDING_COMPONENTS = set()


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
        mode = 'w'
        if os.path.exists(file_path):
            mode = 'a'
        with open(file_path, mode, buffering=1) as f:
            sys.stdout = Tee(original_stdout, f)
            yield
    finally:
        sys.stdout = original_stdout


def _check_secret_health(db):
    try:
        load_secrets()
    except FileNotFoundError as e:
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=HTTP_410_GONE,
            detail=str(e),
        )
    except OSError as e:
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    try:
        check_secrets()
    except IncorrectSecretException as e:
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    try:
        if CFG.data_backend == 'snowflake://':
            load_plugin('snowflake').check_secret_updates(db)
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=HTTP_409_CONFLICT,
            detail=str(e),
        )


def build_rest_app(app: SuperDuperApp):
    """
    Add the key endpoints to the FastAPI app.

    :param app: SuperDuperApp
    """
    CFG.log_colorize = False

    @app.add(
        "/health",
        method="get",
        responses={
            HTTP_410_GONE: {"description": "Secret files are missing"},
            HTTP_401_UNAUTHORIZED: {"description": "Secrets are empty or incorrect"},
            HTTP_409_CONFLICT: {"description": "Secrets are being updated"},
        },
    )
    def health(db: 'Datalayer' = DatalayerDependency()):
        if 'SUPERDUPER_REQUIRED_SECRETS' in os.environ:
            _check_secret_health(db)
        return {"status": 200}

    @app.add("/handshake/config", method="post")
    def handshake(cfg: str):
        from superduper import CFG

        cfg_dict = json.loads(cfg)
        match = CFG.match(cfg_dict)
        if match:
            return {"status": 200, "msg": "matched"}

        diff = CFG.diff(cfg_dict)

        return JSONResponse(
            status_code=400,
            content={"error": f"Config doesn't match based on this diff: {diff}"},
        )

    @app.add('/db/artifact_store/put', method='put')
    def db_artifact_store_put_bytes(
        raw: bytes = File(...), db: 'Datalayer' = DatalayerDependency()
    ):
        file_id = str(hashlib.sha1(raw).hexdigest())
        db.artifact_store.put_bytes(serialized=raw, file_id=file_id)
        return {'file_id': file_id}

    @app.add(
        '/db/artifact_store/get',
        method='get',
    )
    def db_artifact_store_get_bytes(
        file_id: str, db: 'Datalayer' = DatalayerDependency()
    ):
        bytes = db.artifact_store.get_bytes(file_id=file_id)
        if magic is not None:
            media_type = magic.from_buffer(bytes, mime=True)
        else:
            media_type = None
        return Response(content=bytes, media_type=media_type)

    @app.add("/db/upload", method="put")
    def db_upload(raw: bytes = File(...), db: 'Datalayer' = DatalayerDependency()):
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
                    db.artifact_store.put_bytes(content, blob)

            files = os.path.join(temp_dir, "files")
            if os.path.exists(files):
                file_objects = os.listdir(files)
                for file in file_objects:
                    file_path = os.path.join(files, file)
                    db.artifact_store.put_file(file_path, file)

        # TODO add file objects
        # Component to be rendered in the front end
        # Blob objects to be displayed on the upload
        return {"component": component, "artifacts": blob_objects}

    def _process_db_apply(db, component, id: str | None = None):
        def _apply():
            nonlocal component
            variables = None
            build_template = component['build_template']
            identifier = component['identifier']
            if '_variables' in component:
                variables = component['_variables']

            component = Document.decode(component, db=db).unpack()
            if variables:
                component.build_template = build_template
                component.build_variables = variables

            component.identifier = identifier

            db.apply(component, force=True)

        if id:
            log_file = f"/tmp/{id}.log"
            with redirect_stdout_to_file(log_file):
                try:
                    _apply()
                except Exception as e:
                    logging.error(f'Exception during application apply :: {e}')
                    logging.error(traceback.format_exc())
                    PENDING_COMPONENTS.discard(
                        (component.type_id, component.identifier)
                    )
                    raise
        else:
            try:
                _apply()

            except Exception as e:
                logging.error(f'Exception during application apply :: {e}')
                raise

    @app.add('/describe_tables')
    def describe_tables(db: 'Datalayer' = DatalayerDependency()):
        out = db.databackend.list_tables_or_collections()
        return [
            t
            for t in out
            if (
                not t.startswith(CFG.output_prefix)
                and t.lower()
                not in {
                    'component',
                    'job',
                    'parent_child_association',
                    'artifact_relations',
                }
            )
        ]

    @app.add('/db/apply', method='post')
    def db_apply(
        info: t.Dict,
        background_tasks: BackgroundTasks,
        id: str | None = 'test',
        db: 'Datalayer' = DatalayerDependency(),
    ):
        msg = 'Identifier (name) of application should match [a-zA-Z\_0-9]+'
        assert re.match('^[a-zA-Z\_0-9]+$', info['identifier']) is not None, msg

        if 'SUPERDUPER_REQUIRED_SECRETS' in os.environ:
            _check_secret_health(db)

        cls_path = info['_builds'][info['_base'][1:]]['_path']
        cls = import_object(cls_path)
        type_id = cls.type_id
        if (type_id, info['identifier']) in PENDING_COMPONENTS and not db.show(
            type_id, info['identifier']
        ):
            raise Exception(
                f'The component you have added ({type_id}, {info["identifier"]}) '
                'is in the pending state'
            )

        try:
            if db.show(type_id, info['identifier'], -1)['status'] != 'ready':
                raise Exception(
                    f'The component {type_id}:{info["identifier"]} is being processed'
                )
        except FileNotFoundError:
            logging.info(f'Processing a new component {type_id}:{info["identifier"]}')

        PENDING_COMPONENTS.add((type_id, info['identifier']))
        if '_variables' in info:
            info['_variables']['output_prefix'] = CFG.output_prefix
            info['_variables']['databackend'] = db.databackend.backend_name
        background_tasks.add_task(_process_db_apply, db, info, id)
        return {'status': 'ok'}

    import subprocess

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
                    os.remove(filename)
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
        show_status: t.Optional[bool] = False,
        db: 'Datalayer' = DatalayerDependency(),
    ):
        if application is not None:
            r = db.metadata.get_component('application', application)
            return r['namespace']
        else:
            out = db.show(
                type_id=type_id,
                identifier=identifier,
                version=version,
            )
            if show_status:
                assert version is None
                if type_id is not None and identifier is None:
                    out = [{'identifier': x, 'type_id': type_id} for x in out]
                out = [{**r, 'status': 'initialized'} for r in out]
                initialized = [(r['type_id'], r['identifier']) for r in out]
                for pending_app in PENDING_COMPONENTS:
                    if pending_app not in initialized:
                        out.append(
                            {
                                'type_id': pending_app[0],
                                'identifier': pending_app[1],
                                'status': 'pending',
                            }
                        )
            return out

    @app.add('/db/remove', method='post')
    def db_remove(
        type_id: str, identifier: str, db: 'Datalayer' = DatalayerDependency()
    ):
        PENDING_COMPONENTS.discard((type_id, identifier))
        db.remove(type_id=type_id, identifier=identifier, recursive=True, force=True)
        return {'status': 'ok'}

    @app.add('/db/show_template', method='get')
    def db_show_template(
        identifier: str,
        type_id: str = 'template',
        db: 'Datalayer' = DatalayerDependency(),
    ):
        template: Template = db.load(type_id=type_id, identifier=identifier)
        return template.form_template

    @app.add('/db/edit', method='get')
    def db_edit(
        identifier: str,
        type_id: str,
        db: 'Datalayer' = DatalayerDependency(),
    ):
        component = db.load(type_id, identifier)
        template = db.load('template', component.build_template)
        form = template.form_template
        form['_variables'] = component.build_variables
        return form

    @app.add('/db/metadata/show_jobs', method='get')
    def db_metadata_show_jobs(
        type_id: str,
        identifier: t.Optional[str] = None,
        db: 'Datalayer' = DatalayerDependency(),
    ):
        return [
            r['job_id']
            for r in db.metadata.show_jobs(type_id=type_id, identifier=identifier)
            if 'job_id' in r
        ]

    @app.add('/db/execute', method='post')
    def db_execute(query: t.Dict, db: 'Datalayer' = DatalayerDependency()):
        if query['query'].startswith('db.show'):
            output = eval(query["query"])
            logging.info('db.show results:')
            logging.info(output)
            return [{'_base': output}], []

        if '_path' not in query:
            plugin = db.databackend.backend_name
            query['_path'] = f'superduper_{plugin}.query.parse_query'

        q = Document.decode(query, db=db).unpack()

        logging.info('processing this query:')
        logging.info(q)

        result = q.execute()

        if q.type in {'insert', 'delete', 'update'}:
            return {'_base': [str(x) for x in result]}, []

        logging.warn(str(q))

        if isinstance(result, Document):
            result = [result]

        result = [rewrite_artifacts(r, db=db) for r in result]
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
