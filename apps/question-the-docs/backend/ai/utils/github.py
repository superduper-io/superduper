"AI helper functions for loading data from GitHub."

import base64
import json
import os
from pathlib import Path

import requests


# TODO: Use GraphQL API instead of REST API and convert to async
def gh_repo_contents(owner, repo, branch=None):
    def get_repo(branch):
        nonlocal owner, repo
        r = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=true",
            headers={'Authorization': f'token {os.environ["GITHUB_TOKEN"]}'},
        )
        if r.status_code != 200:
            raise Exception(f"Error getting repo contents: {r.status_code, r.json()}")
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
            f"Tried `main` and `master` branches, but neither exist. :: reson {errs}"
        )


def documentation_markdown_urls(repo_contents, documentation_location):
    urls = []
    for val in repo_contents['tree']:
        if documentation_location in val['path'] and val['path'].endswith('.md'):
            urls.append(val['url'])
        else:
            continue
    return urls


def download_and_decode(url):
    blob = requests.get(
        url,
        headers={'Authorization': f'token {os.environ["GITHUB_TOKEN"]}'},
    ).json()
    return base64.b64decode(blob['content'])


def save_github_md_files_locally(repo_details):
    print(f"Downloading files from GitHub for {json.dumps(repo_details)}")
    owner = repo_details['owner']
    name = repo_details['repo']
    branch = repo_details['branch']
    documentation_location = repo_details['documentation_location']

    repo_contents = gh_repo_contents(owner, name, branch)
    urls = documentation_markdown_urls(repo_contents, documentation_location)

    try:
        Path(f"docs/{name}").mkdir(exist_ok=False, parents=True)
    except FileExistsError:
        raise FileExistsError(f"Directory docs/{name} already exists.")

    for i, url in enumerate(urls):
        content = download_and_decode(url)
        with open(f"docs/{name}/file_{i}", 'wb') as f:
            f.write(content)

    return Path(f"docs/{name}").glob("*")


def get_repo_details(path):
    path_split = path.split('/')
    branch = None
    documentation_location = ''
    owner = path_split[3]
    repo = path_split[4]
    if len(path_split) == 7:
        branch = path_split[-1]
    details = {
        'owner': owner,
        'repo': repo,
        'branch': branch,
        'documentation_location': documentation_location,
    }
    return details
