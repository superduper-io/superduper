import { FC, useRef } from "react";
import Answer from "./Answer";
import styled from "styled-components";
interface AiQuestionProp {
  data: any;
  loading: boolean;
  setLoading: any;
  autoScroll: boolean; // Add a prop to control auto-scrolling
  setAutoScroll: any;
  setIsFinish: any;
  collection_name: any;
}

const QuestionTabContainer = styled.div``;

const QuestionTab: FC<AiQuestionProp> = ({
  data,
  loading,
  setLoading,
  autoScroll,
  setAutoScroll,
  setIsFinish,
  collection_name,
}) => {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  return (
    <QuestionTabContainer ref={scrollRef}>
      {data.map((item: any, index: number) => {
        return (
          <Answer
            key={item.id || index}
            index={index}
            item={item}
            setAutoScroll={setAutoScroll}
            setIsFinish={setIsFinish}
            collection_name={collection_name}
          />
        );
      })}
    </QuestionTabContainer>
  );
};

export default QuestionTab;
