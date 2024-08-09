"""
LLM model test cases.
All the llm model can use the check_xxx func to test the intergration with db.
"""

from superduper.base.document import Document
from superduper.components.listener import Listener


def check_predict(db, llm):
    """Test whether db can call model prediction normally."""
    db.add(llm)
    result = llm.predict("1+1=")
    assert isinstance(result, str)


def check_llm_as_listener_model(db, llm):
    """Test whether the model can predict the data in the database normally"""
    collection_name = "question"
    db.cfg.auto_schema = True
    datas = [
        Document({"question": f"1+{i}=", "id": str(i), '_fold': 'train'})
        for i in range(10)
    ]
    db[collection_name].insert(datas).execute()
    select = db[collection_name].select("id", "question")

    listener = Listener(
        select=select,
        key="question",
        model=llm,
    )
    db.add(listener)

    output_select = listener.outputs_select

    results = db.execute(output_select)
    for result in results:
        output = result[listener.outputs_key]
        assert isinstance(output, str)


# TODO: add test for llm_cdc
def check_llm_cdc(db, llm):
    """Test whether the model predicts normally for incremental data"""
    pass


# TODO: Expanded into a test tool class,
# Used to test whether all model objects are normally compatible with superduper
