const sendAllTabs = (filename) => {
    // Select all <li> elements marked as tabs, but only include those that are selected
    const selectedTabs = Array.from(document.querySelectorAll('.tabs > li[role="tab"]'))
                              .filter(tab => tab.getAttribute('aria-selected') === 'true')
                              .map(tab => tab.textContent.trim());

    console.log("About to process filename:", filename);
    console.log("Selected tabs:", selectedTabs);

    const notebookFilename = `_${filename.replace(/\.md$/, '.ipynb')}`;
    const encodedFilename = encodeURIComponent(notebookFilename);
    const postUrl = `https://build-use-cases-sddb.replit.app/build_notebook?usecase_path=.%2Fuse_cases%2F${encodedFilename}`;

    fetch(postUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        body: JSON.stringify(selectedTabs)
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = notebookFilename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        alert('Your file has downloaded!');
    })
    .catch(() => alert('There was an error.'));
    console.log("Sending JSON payload:", JSON.stringify(selectedTabs));
};

export default sendAllTabs;