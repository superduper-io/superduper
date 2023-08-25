import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.2/index.js';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { check} from 'k6';
import http from 'k6/http';

export let options = {
    vus: 20,
    duration: '60s',
};

export default function () {
    let url = 'https://question-the-docs.fly.dev/documents/query';
    let body = JSON.stringify({
        "query": "What is SuperDuperDB and what is its mission?",
        "collection_name": "superduperdb",
    });
    let res = http.post(url, body, {
        headers: {
            'Content-Type': 'application/json',
            'accept': 'application/json',
        },
    });

    check(res, { 'POST status 200': (r) => r.status === 200 });
    console.log(`POST: ${res.status}`)   
}

export function handleSummary(data) {
    const SUMMARY_TITLE = 'qtd-stress';
    let fname = `${SUMMARY_TITLE}.html`;
    let fnameTxt = `${SUMMARY_TITLE}.txt`;

    return {
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
        [fnameTxt]: textSummary(data, { indent: ' ', enableColors: false }),
        [fname]: htmlReport(data, { title: `${SUMMARY_TITLE}` }),
    };
}
