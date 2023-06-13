from pydantic import BaseModel

TYPE_ID = 'type_id'
MODELS = set()
TYPE_ID_TO_MODEL = {}
_NONE = object()


class Serializable(BaseModel):
    def __init_subclass__(cls, *a, **ka):
        super().__init_subclass__(*a, **ka)
        if (type_id := getattr(cls, TYPE_ID, _NONE)) is not _NONE:
            if old_model := TYPE_ID_TO_MODEL.get(type_id):
                raise ValueError(f'Duplicate type_id: old={old_model}, new={cls}')
            TYPE_ID_TO_MODEL[type_id] = cls

        MODELS.add(cls)

    class Config:
        extra = 'forbid'
