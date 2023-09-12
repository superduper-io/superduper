import { useState } from 'react'
import DropdownMenu from './components/DropdownMenu'
import Query from './components/Query'
import Header from './components/Header';
import MarkdownDisplay from './components/MarkdownDisplay';
import SourceUrls from './components/SourceUrls';
import './App.css'


function App() {
  const [responseText, setResponseText] = useState('');
  const [responseURL, setResponseURL] = useState([]);
  const [inputText, setInputText] = useState('');
  const [selectedOption, setSelectedOption] = useState('');

  return (
    <>
      <Header />

      <p className='info-block'>ℹ️ <em>Insert text here.</em></p>

      <DropdownMenu selectedOption={selectedOption} setSelectedOption={setSelectedOption} />

      <Query inputText={inputText} setInputText={setInputText} setResponseText={setResponseText} setResponseURL={setResponseURL} selectedOption={selectedOption} />

      <MarkdownDisplay responseText={responseText} />

      <SourceUrls responseURL={responseURL} />
    </>
  )
}

export default App
