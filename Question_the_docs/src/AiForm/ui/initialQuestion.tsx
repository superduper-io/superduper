import { FC } from "react";
import styled from "styled-components";

interface intialQuestionProps {
  setValue: any;
  addQuestion: any;
  setAutoScroll: any;
  setIsFinish: any;
  collection_name: any;
}

const InitialQuestion = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
`;

const InitialQuestionSuggestions = styled.div`
  display: inline-block;
  padding: 6px 12px;
  background-color: #ecedee;

  color: #000000;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
  text-decoration: none;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;

  &:hover {
    background-color: #d0d0d0;
  }
`;

const IntialQuestions: FC<intialQuestionProps> = ({
  setValue,
  addQuestion,
  setAutoScroll,
  setIsFinish,
  collection_name,

}) => {
  const collectionQuestionsMap: any = {
    superduperdb: [
      "What is SuperDuperDB about?",
      "How can I use SuperDuperDB with MongoDB?",
      "How can I use the OpenAI API on my data in MongoDB with SuperDuperDB?",
    ],
    langchain: [
      "Give an example of an application built with LangChain.",
      "What are Retrievers?",
      "Can you explain what a Prompt template is?",
    ],
    huggingface: [
      "How do I get started?",
      "How do I add a model to Transformers?",
      "Could you summarize new features or models in HuggingFace?",
    ],
  };
  const initialData: any =
    collectionQuestionsMap[collection_name.toLowerCase()] || [];

  const onclick = (v: string) => {
    addQuestion(v);
  };

  return (
    <>
      <InitialQuestion>
        {initialData.map((item: string, index: number) => {
          return (
            <InitialQuestionSuggestions
              key={index}
              onClick={() => {
                collection_name && onclick(item);
                collection_name && setAutoScroll(true);
                setIsFinish(false)
              }}
            >
              {item}
            </InitialQuestionSuggestions>
          );
        })}
      </InitialQuestion>
    </>
  );
};

export default IntialQuestions;
