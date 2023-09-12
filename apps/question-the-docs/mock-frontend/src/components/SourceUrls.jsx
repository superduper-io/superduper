
const SourceUrls = ({ responseURL }) => {
    const headings = responseURL.map(url => url.split('#')[1]);
    return(
    <div className="sources">
        <b>Sources</b>
        <ul>
        {headings.map((item, index) =>
        <li key={index}>
            <a href={responseURL[index]} target="_blank">{item}</a>
        </li>
    )
        }
        </ul>
    </div>
)}

export default SourceUrls;
