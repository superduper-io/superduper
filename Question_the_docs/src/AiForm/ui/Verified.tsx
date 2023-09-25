import { FC } from "react";
import styled from "styled-components";

const VerifiedDiv = styled.div``;

const VerifiedA = styled.a.attrs({
  target: "_blank",
  rel: "noreferrer",
})`
  font-size: 14px !important;
  display: inline-block !important;
  padding: 6px 12px !important;
  background-color: #ecedee !important;
  color: #000000 !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
  text-decoration: none !important;
  border: none !important;
  border-radius: 4px !important;
  cursor: pointer !important;
  transition: background-color 0.3s ease !important;

  &:hover {
    background-color: #d0d0d0 !important;
    color: #000000 !important;
  }
`;

interface VerifiedProp {
  item: string;
  collection_name: any;
}

const Verified: FC<VerifiedProp> = ({ item, collection_name }) => {
  const lastWord = item.split("#")[1] || collection_name;
  return (
    <VerifiedDiv>
      <VerifiedA href={`${item}`}>{lastWord}</VerifiedA>
    </VerifiedDiv>
  );
};

export default Verified;
