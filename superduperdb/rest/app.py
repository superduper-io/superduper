import hashlib
import typing as t

import magic
import yaml
from fastapi import File, Response

from superduperdb import CFG, logging
from superduperdb.base.document import Document
from superduperdb.server import app as superduperapp

assert isinstance(
    CFG.cluster.rest.uri, str
), "cluster.rest.uri should be set with a valid uri"
port = int(CFG.cluster.rest.uri.split(':')[-1])

if CFG.cluster.rest.config is not None:
    try:
        with open(CFG.cluster.rest.config) as f:
            CONFIG = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warn("cluster.rest.config should be set with a valid path")
        CONFIG = {}

app = superduperapp.SuperDuperApp('rest', port=port)


def build_app(app: superduperapp.SuperDuperApp):
    """
    Add the key endpoints to the FastAPI app.

    :param app: SuperDuperApp
    """

    @app.add('/spec/show', method='get')
    def spec_show():
        template_ids = app.db.show(type_id='template')
        tmp = {
            'application': {
                tid: {
                    '_path': f'superduperdb/applications/{tid}',
                    'identifier': {'type': 'str'},
                    **app.db.show(type_id='template', identifier=tid, version=-1)[
                        'info'
                    ],
                }
                for tid in template_ids
            }
        }
        CONFIG['leaves'].update(tmp)
        return CONFIG

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

    @app.add('/db/apply', method='post')
    def db_apply(info: t.Dict):
        component = Document.decode(info).unpack()
        app.db.apply(component)
        return {'status': 'ok'}

    @app.add('/db/remove', method='post')
    def db_remove(type_id: str, identifier: str):
        app.db.remove(type_id=type_id, identifier=identifier, force=True)
        return {'status': 'ok'}

    @app.add('/db/show', method='get')
    def db_show(
        type_id: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
    ):
        return app.db.show(
            type_id=type_id,
            identifier=identifier,
            version=version,
        )

    @app.add('/db/metadata/show_jobs', method='get')
    def db_metadata_show_jobs(type_id: str, identifier: t.Optional[str] = None):
        return [
            r['job_id']
            for r in app.db.metadata.show_jobs(
                type_id=type_id, component_identifier=identifier
            )
            if 'job_id' in r
        ]

    @app.add('/db/execute', method='post')
    def db_execute(
        query: t.Dict,
    ):
        if '_path' not in query:
            databackend = app.db.databackend.__module__.split('.')[-2]
            query['_path'] = f'superduperdb/backends/{databackend}/query/parse_query'

        q = Document.decode(query, db=app.db).unpack()

        logging.info('processing this query:')
        logging.info(q)

        result = q.execute()

        if q.type in {'insert', 'delete', 'update'}:
            return {'_base': [str(x) for x in result[0]]}, []

        logging.warn(str(q))

        if isinstance(result, Document):
            result = [result]

        result = [r.encode() for r in result]
        result = [(dict(r), list(r.pop_blobs().keys())) for r in result]
        return result


build_app(app)
