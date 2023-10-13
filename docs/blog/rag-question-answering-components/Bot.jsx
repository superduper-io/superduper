import React, {useState} from "react"
import ReactMarkdown from 'react-markdown'
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {hopscotch} from 'react-syntax-highlighter/dist/esm/styles/prism';

const MarkdownDisplay = ({responseText}) => {
    return (
        <div className="box-border">    
            <ReactMarkdown
            children={responseText} 
            components={{
                code({node, inline, className, children, ...props}) {
                const match = /language-(\w+)/.exec(className || '')
                return !inline && match ? (
                    <SyntaxHighlighter
                    {...props}
                    children={String(children).replace(/\n$/, '')}
                    style={hopscotch}
                    language={match[1]}
                    PreTag="div"
                    />
                ) : (
                    <code {...props} className={className}>
                    {children}
                    </code>
                )
                }
            }}
            />        
        </div>
        )
    };

    const handleSubmit = async ({ inputText, setResponseText }) => {
      try {
        setResponseText('Awaiting response to "' + inputText + '"...');
        const response = await fetch('https://question-the-docs.fly.dev/documents/query', {
          method: 'POST',
          headers: {
            'accept': 'application/json',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ "query": inputText, "collection_name": 'superduperdb' }),
        });
        const data = await response.json();
        setResponseText(data.answer);
      } catch (error) {
        console.error('Error:', error);
      }
    };

    const Query = ({ inputText, setInputText, setResponseText, question}) => {

      const handleInputChange = (event) => {
          setInputText(event.target.value);
      };
  
      const submit = async () => {await handleSubmit({inputText, setResponseText})};
  
      return (
          <div>
          <input type="text" placeholder={question} value={inputText} onChange={handleInputChange} size="25" />
          <button className='submit' onClick={submit}>Submit</button>
          </div>
      )
  };


function Bot({question, answer}) {
  const [responseText, setResponseText] = useState(answer);
  const [inputText, setInputText] = useState('');

  return (
    <>
    <em>
    <Query inputText={inputText} setInputText={setInputText} setResponseText={setResponseText} question={question} />
    <br></br>
    <MarkdownDisplay responseText={responseText} />
    </em>
    </>
  )
}

export default Bot
