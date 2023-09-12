const handleSubmit = async ({ inputText, setResponseText, setResponseURL, selectedOption }) => {
    try {
      setResponseText('Awaiting response to "' + inputText + '"...');
      setResponseURL([]);
      const response = await fetch('http://localhost:8000/documents/query', {
        method: 'POST',
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "query": inputText, "collection_name": selectedOption }),
      });
      const data = await response.json();
      setResponseText(data.answer);
      setResponseURL(data.source_urls)
    } catch (error) {
      console.error('Error:', error);
    }
  };

  export default handleSubmit;
  