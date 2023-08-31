const handleSubmit = async ({ inputText, setResponseText, setResponseURL, selectedOption }) => {
    try {
      setResponseText('Awaiting response to "' + inputText + '"...');
      setResponseURL([]);
      
      const streamResponse = await fetch('http://localhost:8000/documents/vector-search/summary', {
        method: 'POST',
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "query": inputText, "collection_name": selectedOption }),
      });
      const reader = streamResponse.body.pipeThrough(new TextDecoderStream()).getReader();
      
      let currentAnswer = '';
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        currentAnswer += value;
        setResponseText(currentAnswer);
      }
      
      const sourceResponse = await fetch('http://localhost:8000/documents/vector-search', {
        method: 'POST',
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "query": inputText, "collection_name": selectedOption }),
      });
      
      const source = await sourceResponse.json();
      setResponseURL(source.urls)
    
    } catch (error) {
      console.error('Error:', error);
    }
  };

  export default handleSubmit;
  