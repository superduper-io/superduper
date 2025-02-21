import os
import sys

from superduper import Schema, Table
from superduper.base.datatype import file


pdf_folder = sys.argv[1]
pdf_names = [pdf for pdf in os.listdir(pdf_folder) if pdf.endswith(".pdf")]
pdf_paths = [os.path.join(pdf_folder, pdf) for pdf in pdf_names]
data = [{"url": pdf_path, "file": pdf_path} for pdf_path in pdf_paths]
from superduper import superduper
db = superduper()
COLLECTION_NAME = next(x for x in pdf_folder.split('/')[::-1] if x)
schema = Schema(identifier="myschema", fields={'url': 'str', 'file': file})
table = Table(identifier=COLLECTION_NAME, schema=schema)
db.apply(table, force=True)
db[COLLECTION_NAME].insert(data).execute()