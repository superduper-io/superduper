import dataclasses as dc
import re

T_NOT_IN_DOCUMENT = "&"
T_IN_DOCUMENT = "?"


@dc.dataclass
class Reference:
    """A reference to a database or a leaf.

    :param type: The type of the reference. Either "&" for a database or "?" for a leaf.
    :param name: The name of the reference.
    :param path: The path of the reference, e.g. type_id:identifier:version
    """

    type: str
    name: str
    path: str

    @property
    def is_in_document(self):
        """Check if the reference is in the document."""
        return self.type == T_IN_DOCUMENT

    def __bool__(self):
        return bool(self.type) and bool(self.name)


def parse_reference(string) -> Reference:
    """Parse a string into a Reference object.

    :param string: The string to parse.
    """
    if not isinstance(string, str):
        return Reference("", "", "")

    match = re.match(r"^(\?|&):(.*?):(.*)$", string)

    if not match:
        return Reference("", "", "")

    return Reference(match.group(1), match.group(2), match.group(3))
