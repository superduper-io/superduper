import axios from "axios";
import styled from "styled-components";
import QuestionTab from "./QuestionTab";
import IntialQuestions from "./initialQuestion";
import { useEffect, useRef, useState } from "react";
import useOnClickOutside from "../../customHook/useOnClickOutside";

let apikey: any;
const data: any = document?.getElementById("my-api") || null;
apikey = data?.getAttribute("data-api-key") || "superduperdb";

const Container = styled.div`
  display: flex !important;
  padding-inline: 50px !important;
  justify-content: end !important;
  & * {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
      Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue",
      sans-serif !important;
  }
  @media (max-width: 768px) {
    justify-content: center !important;
  }
`;

const Card = styled.div`
  display: flex !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
  border-radius: 12px !important;
  flex-wrap: wrap !important;
  background-color: white;
  color: black !important;
  width: 55vw !important;
  position: fixed !important;
  z-index: 2147483647 !important;
  bottom: 20px !important;
  padding: 45px 1.5rem 18px !important;
  justify-content: center !important;
  gap: 14px !important;
  font-size: 16px !important;

  @media (max-width: 767px) {
    width: 85vw !important;
  }
  @media (min-width: 768px) {
    width: 55vw !important;
  }
  box-shadow: 0px 0px 8px 2px #7628f8;
`;

const Overflow = styled.div`
  padding-right: 25px !important;
  max-height: 300px !important;
  overflow: auto !important;
  line-height: 24px !important;
  text-align: justify !important;
  padding-top: 1rem !important;
  margin-bottom: 30px;
  ::-webkit-scrollbar {
    width: 5px !important;
    height: 5px !important;
  }
  ::-webkit-scrollbar-thumb {
    border-radius: 10px !important;
    background: #888 !important;
  }
`;

const CardBody = styled.div`
  margin-bottom: 27px !important;
`;

const SvgForm = styled.svg.attrs({
  xmlns: "http://www.w3.org/2000/svg",
  xmlnsXlink: "http://www.w3.org/1999/xlink",
  viewBox: "0 0 489.533 489.533",
  width: "9px",
})``;

const ResetButton = styled.button`
  display: block;
  font-weight:900;
  position: absolute !important;
  height: 25px !important;
  width: 25px !important;
  font-size: 14px !important;
  right: 38px !important;
  top: 5px !important;
  padding-left: 6.5px !important;
  cursor: pointer !important;
  background-color: #ecedee;
  border: none !important;
  border-radius: 50% !important; /* Make it round */
  transition: background-color 0.3s ease;
  &:hover {
    background-color: #d0d0d0;
  }
`;

const CloseButton = styled.button`
  display: block;
  position: absolute !important;
  height: 25px !important;
  width: 25px !important;
  font-size: 10px !important;
  right: 7px !important;
  top: 5px !important;
  padding: 0px !important;
  cursor: pointer !important;
  background-color: #ecedee;
  border: none !important;
  border-radius: 50% !important; /* Make it round */
  transition: background-color 0.3s ease;
  &:hover {
    background-color: #d0d0d0;
  }
`;

const ResetButtonSpan = styled.span``;

const TextInput1 = styled.input`
  width: 254px;
  right:20px;
  padding-inline: 20px !important;
  font-size: 16px !important;
  height: 42px !important;
  border-radius: 30px !important;
  border: none !important;
  outline: 0.2px solid lightgray !important;
  position: fixed !important;
  bottom: 0 !important;
  background: white !important;
  transform: translate(0, -35%) !important;
  box-shadow: 0px 0px 8px 2px #7628f8;
`;

const TextInput2 = styled.input`
  position: fixed !important;
  bottom: 0 !important;
  transform: translate(0, -150%) !important;
  width: 54% !important;
  padding-inline: 20px !important;
  font-size: 16px !important;
  background: white !important;
  height: 42px !important;
  border-radius: 30px !important;
  border: none !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
  border-radius: 30px !important;
  outline: 0.2px solid lightgray !important;
  // -webkit-transition: 0.5s ease-in-out;
  // transition: 0.5s ease-in-out !important;
  @media (max-width: 768px) {
    width: 82% !important;
  }
`;

const Intro = styled.div`
  display: flex !important;
  gap: 1rem !important;
  padding-bottom: 2rem !important;
  justify-content: center !important;
`;

const Emoji = styled.div`
  font-size: 28px !important;
  margin-right: 2px !important;
`;

const InitialMessage = styled.div`
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
`;

const PoweredBy = styled.div`
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
  font-size: 12px !important;
  width: 100% !important;
`;

const PoweredByA = styled.a.attrs({
  href: "https://superduperdb.com/",
  target: "_blank",
  rel: "noreferrer",
})`
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
  text-decoration: underline !important;
  color: #7628f8 !important;
  text-decoration-color: #7628f8 !important;
`;

const HereA = styled.a.attrs({
  href: "https://www.superduperdb.com/blog",
  target: "_blank",
  rel: "noreferrer",
})`
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
  text-decoration: underline !important;
  color: #7628f8 !important;
  text-decoration-color: #7628f8 !important;
`;

const AiForm: any = () => {
  const [data, setData] = useState<any>([]);
  const [vis, setVis] = useState(false);
  const [value, setValue] = useState<any>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [autoScroll, setAutoScroll] = useState<boolean>(false);
  const [isFinish, setIsFinish] = useState<boolean>(true);
  const [collection_name, setCollectionName] = useState(`${apikey}`);
  const abortControllerRef = useRef<any>(null);
  const enterRef = useRef<any>();

  const popupRef = useRef<any>();
  const cardBodyRef = useRef<any>();

  const addQuestion: any = async (value: string) => {
    setValue("");
    try {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const abortController = new AbortController();
      abortControllerRef.current = abortController;
      let questionData: any;

      setData([
        ...data,
        {
          question: value,
          answer: "",
          verified_source: [],
        },
      ]);
      if (collection_name) {
        const axiosPromise = axios.post(
          "https://question-the-docs.fly.dev/documents/query",
          {
            query: value,
            collection_name: collection_name.toLowerCase(),
          },
          {
            signal: abortController.signal,
          }
        );
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => {
            abortController.abort();
            reject(
              new Error(
                "Sorry but the OpenAI API is at capacity. Please try again."
              )
            );
          }, 30000);
        });

        questionData = await Promise.race([axiosPromise, timeoutPromise]);
      }

      const newQuestionAnswer = [
        ...data,
        {
          question: value,
          answer: "",
          verified_source: [],
        },
      ];
      const answer = questionData?.data;

      const newData = newQuestionAnswer.map((item) => {
        if (item.question === value) {
          return {
            question: value,
            answer: answer?.answer || "Documentation not found",
            verified_source: answer?.source_urls || [],
          };
        } else {
          return item;
        }
      });

      setData([...newData]);
    } catch (error: any) {
      if (error.message === "canceled") {
        setData([]);
      } else {
        setData([
          ...data,
          {
            question: value,
            answer: error.message,
            verified_source: [],
          },
        ]);
      }
    }
  };

  useEffect(() => {
    const scrollDown = () => {
      if (cardBodyRef.current && autoScroll) {
        cardBodyRef.current.scrollTop += 10;
      }
    };
    const scrollInterval: any = autoScroll && setInterval(scrollDown, 18);

    let lastScrollTop = 0;

    const handleScroll = () => {
      if (autoScroll) {
        const scrollTop = cardBodyRef.current.scrollTop;
        const scrollDelta = scrollTop - lastScrollTop;

        if (scrollDelta >= -20) {
          lastScrollTop = scrollTop;
        } else {
          clearInterval(scrollInterval);
        }
      }
    };

    if (cardBodyRef.current) {
      cardBodyRef.current.addEventListener("scroll", handleScroll);
    }

    return () => {
      if (cardBodyRef.current) {
        cardBodyRef.current.removeEventListener("scroll", handleScroll);
      }
      clearInterval(scrollInterval);
    };
  }, [autoScroll]);

  useOnClickOutside(popupRef, () => {
    setVis(false);
  });

  function handleClick() {
    setVis(true);
  }

  return (
    <>
      <Container ref={popupRef}>
        <Card style={{ background: vis ? "" : "transparent", boxShadow: vis ? "" : "none" }}>
          <ResetButton
            style={{ display: vis ? "" : "none" }}
            onClick={() => {
              setData([]);
              setAutoScroll(false);
              setIsFinish(true);
              if (abortControllerRef.current) {
                abortControllerRef.current.abort();
              }
            }}
          >
            ⟳
          </ResetButton>
          <CloseButton
            style={{ display: vis ? "" : "none" }}
            onClick={() => {
              setVis(false);
            }}
          >
            ❌
          </CloseButton>
          <CardBody style={{ display: vis ? "block" : "none" }}>
            <Overflow ref={cardBodyRef}>
              <Intro>
                <Emoji>🔮</Emoji>
                <InitialMessage>
                  {" "}
                  Hi! I am an AI chatbot and you can ask me anything. If the
                  documentation contains sufficient information I will be able
                  to provide an answer 🙂. If you want to read more about how I
                  was built, read more <HereA>here</HereA>.
                </InitialMessage>
              </Intro>
              {data.length ? (
                <QuestionTab
                  setLoading={setLoading}
                  loading={loading}
                  data={data}
                  collection_name={collection_name}
                  autoScroll={autoScroll}
                  setAutoScroll={setAutoScroll}
                  setIsFinish={setIsFinish}
                />
              ) : (
                <IntialQuestions
                  addQuestion={addQuestion}
                  setValue={setData}
                  setAutoScroll={setAutoScroll}
                  setIsFinish={setIsFinish}
                  collection_name={collection_name}
                />
              )}
            </Overflow>
          </CardBody>
          {vis ? (
            <TextInput2
              id="input"
              ref={enterRef}
              onFocus={handleClick}
              type="text"
              value={value}
              onChange={(e) => {
                isFinish && setValue(e.target.value);
              }}
              disabled={loading && isFinish && vis}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  isFinish && value && addQuestion(value);
                  collection_name && value && setIsFinish(false);
                  isFinish && value && setAutoScroll(true);
                }
              }}
              placeholder="Enter your question or request here!"
            />
          ) : (
            <TextInput1
              id="input"
              onFocus={handleClick}
              type="text"
              placeholder="Enter your question or request here!"
            />
          )}

          <PoweredBy style={{ display: vis ? "block" : "none" }}>
            Built with&nbsp;
            <PoweredByA>SuperDuperDB</PoweredByA>
          </PoweredBy>
        </Card>
      </Container>
    </>
  );
};

export default AiForm;
