"""
LLM model test cases.
All the llm model can use the check_xxx func to test the intergration with db.
"""

from superduper.base.base import Base
from superduper.base.document import Document
from superduper.components.listener import Listener


def check_predict(db, llm):
    """Test whether db can call model prediction normally."""
    db.apply(llm)
    result = llm.predict("1+1=")
    assert isinstance(result, str)


class question(Base):
    id: str
    _fold: str
    question: str


def check_llm_as_listener_model(db, llm):
    """Test whether the model can predict the data in the database normally"""
    collection_name = "question"

    db.create(question)

    datas = [
        Document({"question": f"1+{i}=", "id": str(i), '_fold': 'train'})
        for i in range(10)
    ]
    db[collection_name].insert(datas)
    select = db[collection_name]

    listener = Listener(
        identifier="listener",
        select=select,
        key="question",
        model=llm,
    )
    db.apply(listener)

    results = db[listener.outputs].select().execute()
    for result in results:
        output = result[listener.outputs]
        assert isinstance(output, str)


# TODO: add test for llm_cdc
def check_llm_cdc(db, llm):
    """Test whether the model predicts normally for incremental data"""
    pass


# TODO: Expanded into a test tool class,
# Used to test whether all model objects are normally compatible with superduper
