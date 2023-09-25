import React from "react";
import * as ReactDOM from "react-dom";
import { StyleSheetManager } from "styled-components";

function parseValue(value: any) {
  if (value === "" || value === "true") {
    return true;
  }

  if (value === "false") {
    return false;
  }

  if (Number(value).toString() === value) {
    return Number(value);
  }

  return value;
}

function parseKey(key: string) {
  const parts = key.split("-");
  const newParts = [parts[0]];
  for (let i = 1; i < parts.length; i++) {
    const firstLetter = parts[i].slice(0, 1);
    const restOfLetters = parts[i].slice(1);
    const newPart = firstLetter.toUpperCase() + restOfLetters;
    newParts.push(newPart);
  }
  return newParts.join("");
}

function attrToObj(attrs: NamedNodeMap) {
  const attrsObj: { [key: string]: unknown } = {};
  const length = attrs.length;
  for (let i = 0; i < length; i++) {
    const { name, value } = attrs[i];
    attrsObj[parseKey(name)] = parseValue(value);
  }
  return attrsObj;
}

async function HtmlTagWrapper(Component: (props?: any) => JSX.Element) {
  const el: any = document.getElementById("superduperdb_app");
  const shadowRoot = el.attachShadow({ mode: "open" });
  const styleSlot = document.createElement("section");
  shadowRoot.appendChild(styleSlot);

  const renderIn = document.createElement("div");

  styleSlot.appendChild(renderIn);
  const attrs = el.attributes;
  const props = attrToObj(attrs);
  console.log(props);
  ReactDOM.render(
    <StyleSheetManager target={styleSlot}>
      <Component {...props} />
    </StyleSheetManager>,
    renderIn
  );
}

export { HtmlTagWrapper };
