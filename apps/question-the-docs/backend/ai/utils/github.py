'AI helper functions for loading data from GitHub.'

import base64
import dataclasses as dc
import os
import re
from pathlib import Path

import requests

URL_CACHE = {}


@dc.dataclass
class Repo:
    owner: str
    name: str
    branch: str
    documentation_dir: str
    documentation_base_url: str
    documentation_file_extension: str = 'html'


superduperdb = Repo(
    'SuperDuperDB',
    'superduperdb',
    'main',
    'docs/',
    'https://superduperdb.github.io/superduperdb',
)
langchain = Repo(
    'langchain-ai', 'langchain', 'master', 'docs/docs_skeleton/docs', 'https://python.langchain.com/docs'
)


huggingface= Repo(
    'huggingface', 'transformers', 'main', 'docs/source/en', 'https://huggingface.co/docs/transformers', documentation_file_extension=''
)


REPOS = {'superduperdb': superduperdb, 'langchain': langchain, 'huggingface': huggingface}


# TODO: Use GraphQL API instead of REST API and convert to async
def gh_repo_contents(owner, repo, branch=None):
    def get_repo(branch):
        nonlocal owner, repo
        r = requests.get(
            f'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=true',
            headers={'Authorization': f'token {os.environ["GITHUB_TOKEN"]}'},
        )
        if r.status_code != 200:
            raise Exception(f'Error getting repo contents: {r.status_code, r.json()}')
        return r.json()

    if branch:
        return get_repo(branch)
    else:

        errs = []
        for br in ['main', 'master']:
            try:
                return get_repo(br)
            except Exception as e:
                errs.append(e)
                continue
        raise Exception(
            f'Tried `main` and `master` branches, but neither exist. Reason: {errs}'
        )


def documentation_markdown_urls(repo_contents, documentation_dir):
    urls = []
    for val in repo_contents['tree']:
        if documentation_dir in val['path'] and val['path'].endswith(('.md', '.mdx')):
            urls.append(val)
        else:
            continue
    return urls


def download_and_decode(url):
    blob = requests.get(
        url,
        headers={'Authorization': f'token {os.environ["GITHUB_TOKEN"]}'},
    ).json()
    return base64.b64decode(blob['content'])


def save_github_md_files_locally(repo):
    try:
        repo_metadata = REPOS[repo]
    except KeyError:
        raise Exception(f'Repo {repo} not found in {REPOS.keys()}')

    owner = repo_metadata.owner
    name = repo_metadata.name
    branch = repo_metadata.branch
    documentation_dir = repo_metadata.documentation_dir
    documentation_base_url = repo_metadata.documentation_base_url
    doc_extension = repo_metadata.documentation_file_extension

    repo_contents = gh_repo_contents(owner, name, branch)
    urls = documentation_markdown_urls(repo_contents, documentation_dir)

    Path(f'docs/{repo}').mkdir(exist_ok=True, parents=True)

    URL_CACHE[(repo, '')] = documentation_base_url

    def doc_base_path(path, section, ext):
        if ext:
            return f'{documentation_base_url}/{path}.{ext}#{section}'
        else:
            return f'{documentation_base_url}/{path}#{section}'


    def _remove_overlap_prefix(x, y):
        x_dirs = x.split('/')
        y_dirs  = y.split('/')
        for i in range(len(y_dirs)):
            if y_dirs[i] == x_dirs[-i]:
                return '/'.join(y_dirs[i+1:])
        return y


    for i, url in enumerate(urls):
        content = download_and_decode(url['url'])
        sections = re.findall(r'^\s*(#+)\s*(.*)', content.decode('utf-8'), re.MULTILINE)
        for _, s in sections:
            relative_path = url['path']
            relative_path = '/'.join(relative_path.split('/')[1:])
            relative_path = _remove_overlap_prefix(documentation_dir, relative_path)
            section_file = os.path.splitext(relative_path)[0]
            s_encoded = s.replace(' ', '-').lower()
            if documentation_base_url:
                URL_CACHE[(repo, s)] = doc_base_path(section_file, s_encoded, doc_extension)
            else:
                URL_CACHE[(repo, s)] = relative_path
        with open(f'docs/{repo}/file_{i}', 'wb') as f:
            f.write(content)
    return Path(f'docs/{repo}').glob('*')
