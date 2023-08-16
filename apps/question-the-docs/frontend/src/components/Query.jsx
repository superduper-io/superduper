import handleSubmit from '../services/queries'

const Query = ({ inputText, setInputText, setResponseText, setResponseURL, selectedOption}) => {

    const handleInputChange = (event) => {
        setInputText(event.target.value);
    };

    const submit = async () => {await handleSubmit({inputText, setResponseText, setResponseURL, selectedOption})};

    return (
        <div>
        <input type="text" placeholder="Ask a question" value={inputText} onChange={handleInputChange}/>
        <button className='submit' onClick={submit}>Submit</button>
        </div>
    )
};

export default Query;