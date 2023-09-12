import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { hopscotch } from 'react-syntax-highlighter/dist/esm/styles/prism';

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

export default MarkdownDisplay;
