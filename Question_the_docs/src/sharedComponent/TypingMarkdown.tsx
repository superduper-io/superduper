import styled from "styled-components";
import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface TypingMarkdownProps {
  source: string;
  speed?: number;
  setAutoScroll: (value: boolean) => void;
  setAnswerFinish: (value: boolean) => void;
  setIsFinish: (value: boolean) => void;
}

const TypingMarkdown: React.FC<TypingMarkdownProps> = ({
  source,
  speed = 1,
  setAutoScroll,
  setAnswerFinish,
  setIsFinish,
}) => {
  const [typedText, setTypedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const typeNextCharacter = () => {
      if (currentIndex < source.length) {
        setTypedText((prevText) => prevText + source[currentIndex]);
        setCurrentIndex(currentIndex + 1);
      } else {
        setAutoScroll(false);
        setAnswerFinish(true);
        setIsFinish(true);
      }
    };

    const typingTimeout = setTimeout(typeNextCharacter, speed);

    return () => clearTimeout(typingTimeout);
  }, [
    currentIndex,
    speed,
    source,
    setAutoScroll,
    setAnswerFinish,
    setIsFinish,
  ]);

  return (
    <ReactMarkdown components={{ code: (props) => <CodeBlock {...props} /> }}>
      {typedText}
    </ReactMarkdown>
  );
};

interface CodeBlockProps {
  node?: any;
  inline?: any;
  className?: any;
  children?: any;
}

const CodeBlockContainer = styled.div`
align-items: center !important;
background-color: #f4f4f4 !important;
border-radius: 8px !important;
margin-top: 10px !important;
margin-bottom: 10px !important;
overflow-x: auto !important;
font-size: 15px !important;
color: #333 !important;
@media (max-width: 768px) {
  max-width: 70vw !important;
}
@media (min-width: 768px) {
  max-width: 44vw !important;
}
`;

const CodeBlockSpan = styled.span`
background-color: #282c34 !important;
color: #61dafb !important;
padding: 8px !important;
border-radius: 5px !important;
font-size: 14px !important;
font-family: monospace !important;
white-space: pre-wrap !important;
line-height: 1.5 !important;
box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1) !important;
cursor: text !important;
transition: background-color 0.3s !important;
display: inline-block !important;
margin: 5px !important;

&::selection {
  background-color: lilac !important;
  color: black !important;
}
`;

const CodeBlock: React.FC<CodeBlockProps> = ({
  node,
  inline,
  className,
  children,
  ...props
}) => {
  const isCodeBlock =typeof children[0] === "string" && children[0].endsWith("\n");

  if (isCodeBlock) {
    return (
      <CodeBlockContainer>
        <SyntaxHighlighter language="python" style={a11yDark}>
          {children}
        </SyntaxHighlighter>
      </CodeBlockContainer>
    );
  } else {
    return <CodeBlockSpan>{children}</CodeBlockSpan>;
  }
};

export default TypingMarkdown;
