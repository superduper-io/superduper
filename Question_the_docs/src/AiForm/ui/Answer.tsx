import { FC, useEffect, useState } from "react";
import Verified from "./Verified";
import TypingMarkdown from "../../sharedComponent/TypingMarkdown";
import styled from "styled-components";

const AnswerMainDiv = styled.div``;

const UserQuestion = styled.div`
  display: flex !important;
  gap: 1.5rem !important;
  padding: 1.5rem 0 !important;
  border-top: 0.5px solid lightgrey !important;
  border-bottom: 0.5px solid lightgrey !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
`;

const ChatUser = styled.span`
  font-size: 28px !important;
`;

const Loader = styled.div`
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  margin-top: 0.5rem !important;
`;

const LoaderSpan = styled.span`
  font-size: 30px;
  animation: spin 1s linear infinite;
  opacity: 0;
  @keyframes spin {
    0% {
      transform: scale(0.5);
    }

    100% {
      opacity: 1;
      transform: none;
    }
  }
`;

const LoaderText = styled.div`
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
  color: rgb(108, 117, 125) !important;
  font-size: 14px !important;
  text-align: center !important;
`;

const AnswerCodeContainer = styled.div`
  text-align: left !important;
  display: flex !important;
  gap: 1rem !important;
  margin-top: 1rem !important;
  div {
    p {
      margin: 0 !important;
    }
  }
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
`;

const AnswerCodeContainerSpan = styled.span`
  font-size: 28px !important;
  margin-right: 2px !important;
`;

const MainVerifiedSource = styled.div`
  padding-inline-start: 3rem !important;
  margin-bottom: 3rem !important;
`;

const MainVerifiedSourceP = styled.p`
  font-size: 14px !important;
  border-top: 0.5px solid !important;
  padding-top: 10px !important;
  font-weight: 500 !important;
  margin-bottom: 20px !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
`;

const VerifiedResponse = styled.div`
  display: flex !important;
  flex-wrap: wrap !important;
  column-gap: 1rem !important;
  row-gap: 0.4rem !important;
  margin-bottom: -2rem !important;
`;

const AnswerResponseContainer = styled.div`
  margin-bottom: 20px !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif ;
  & *:not(pre) {
    font-family: inherit !important; /* Inherit the font family from the parent for all elements except <pre> */
  }
`;

const QuestionContainer = styled.span`
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
`;

interface AiQuestionProp {
  item: any;
  index: any;
  setAutoScroll: any;
  setIsFinish: any;
  collection_name: any;
}

const Answer: FC<AiQuestionProp> = ({
  item,
  index,
  setAutoScroll,
  setIsFinish,
  collection_name,
}) => {
  const [loading, setLoading] = useState(true);
  const [answerFinish, setAnswerFinish] = useState(false);

  useEffect(() => {
    if (item.answer) {
      setLoading(false);
    }
  }, [item.answer]);

  return (
    <AnswerMainDiv key={index}>
      <UserQuestion>
        <ChatUser>👤</ChatUser>
        <QuestionContainer> {item.question}</QuestionContainer>
      </UserQuestion>
      {loading ? (
        <>
          <Loader>
            <LoaderSpan>🔮</LoaderSpan>
          </Loader>
          <LoaderText>
            Uno momento, let me read through the docs real quick!
          </LoaderText>
        </>
      ) : (
        <>
          <AnswerCodeContainer>
            <AnswerCodeContainerSpan>🔮</AnswerCodeContainerSpan>
            <AnswerResponseContainer>
              <TypingMarkdown
                source={item.answer}
                setAutoScroll={setAutoScroll}
                setAnswerFinish={setAnswerFinish}
                speed={0.1}
                setIsFinish={setIsFinish}
              />
            </AnswerResponseContainer>
          </AnswerCodeContainer>

          {answerFinish &&
          item.verified_source &&
          item.verified_source.length > 0 ? (
            <MainVerifiedSource>
              <MainVerifiedSourceP>Verified Sources: </MainVerifiedSourceP>
              <VerifiedResponse>
                {item.verified_source.map(
                  (verifiedItem: any, index: number) => {
                    return (
                      <Verified
                        collection_name={collection_name}
                        key={index}
                        item={verifiedItem}
                      />
                    );
                  }
                )}
              </VerifiedResponse>
            </MainVerifiedSource>
          ) : null}
        </>
      )}
    </AnswerMainDiv>
  );
};
export default Answer;
