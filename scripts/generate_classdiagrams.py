#!/usr/bin/env python3

""" Script to map the pyreverse DSL to the mermaid DSL to generate classDiagram.

Ref:
- https://www.bhavaniravi.com/python/generate-uml-diagrams-from-python-code
- https://mermaid.js.org/syntax/classDiagram.html

Usage example:

cd superduperdb
pyreverse -A -m y -k .
cat packages.dot | ./../scripts/generate_classdiagrams.py > packages.mermaid
cat classes.dot | ./../scripts/generate_classdiagrams.py > classes.mermaid
"""

import re
import sys


def test__defined_arrow():
    tests = [
        {
            "input": """"foo" -> "bar.qux" [arrowhead="open", arrowtail="none"];""",
            "want": "-->",
        },
        {
            "input": """"foo" -> "bar.qux" [arrowhead="diamond", arrowtail="none"];""",
            "want": "--*",
        },
        {
            "input": """"foo" -> "bar.qux" [arrowhead="empty", arrowtail="none"];""",
            "want": "--|>",
        },
        {
            "input": """"foo" -> "bar.qux" [arrowhead="odiamond", arrowtail="none"];""",
            "want": "--o",
        }
    ]

    for test in tests:
        assert test["want"] == _defined_arrow(test["input"])


def _defined_arrow(el: str) -> str:
    regex = re.compile("""arrowhead="(\\w+)""")

    o = "--"

    arrow_definition = regex.findall(el)[0]
    if arrow_definition == "open":
        o += ">"
    elif arrow_definition == "diamond":
        o += "*"
    elif arrow_definition == "empty":
        o += "|>"
    elif arrow_definition == "odiamond":
        o += "o"

    return o


def _clean(el: str) -> str:
    return el.split(" [")[0].replace("->", _defined_arrow(el)).replace(".", "-").replace("\"", "")


def _is_relation(el: str) -> bool:
    return "->" in el


def test_main():
    tests = [
        {
            "input": """digraph "packages" {""",
            "want": "",
        },
        {
            "input": """rankdir=BT""",
            "want": "",
        },
        {
            "input": "charset=\"utf-8\"",
            "want": "",
        },
        {
            "input": """"foo" [color="aliceblue", label=<foo>, shape="box", style="filled"];""",
            "want": "",
        },
        {
            "input": """"bar.qux" [color="aliceblue", label=<bar>, shape="box", style="filled"];""",
            "want": "",
        },
        {
            "input": """"foo" -> "bar.qux" [arrowhead="open", arrowtail="none"];""",
            "want": "foo --> bar-qux",
        },
    ]

    for test in tests:
        assert test["want"] == main(test["input"])


def main(line: str) -> str:
    """
    Main runner.

    :param line: Input pyreverse DSL's line.
    :returns Mermaid DSL.
    """
    return _clean(line) if _is_relation(line) else ""


if __name__ == "__main__":
    tag = "classDiagram"
    print(tag)

    for line in sys.stdin:
        _in = line.rstrip()
        if 'q' == _in:
            break

        _out = main(_in)
        if _out != "":
            print(_out)
