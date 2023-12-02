from superduperdb.components.encoder import Encoder

int_type = Encoder(
    identifier='int',
    encoder=lambda x: int(x),
)
float_type = Encoder(
    identifier='float',
    encoder=lambda x: float(x),
)
str_type = Encoder(
    identifier='str',
    encoder=lambda x: str(x),
)

bool_type = Encoder(
    identifier='bool',
    encoder=lambda x: bool(x),
)
