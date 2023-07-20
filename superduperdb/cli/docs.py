from . import command
from superduperdb import ROOT
from superduperdb.misc import run
from superduperdb.misc.git import Git, git
from typer import Argument, Option
import functools
import shutil

DOCS = 'docs'
DOCS_ROOT = ROOT / DOCS
SOURCE_ROOT = DOCS_ROOT / 'source'
CODE_ROOT = ROOT / 'superduperdb'
GH_PAGES = ROOT / '.cache/gh-pages'
UPSTREAM = 'git@github.com:SuperDuperDB/superduperdb-stealth.git'

run_gh = functools.partial(run.run, cwd=GH_PAGES)
out_gh = functools.partial(run.out, cwd=GH_PAGES)
git_gh = Git(out=run_gh)


@command(help='Build documentation, optionally committing and pushing got ')
def docs(
    commit_message: str = Argument(
        '',
        help=(
            'The git commit message for the docs update.'
            'An empty message means do not commit and push.'
        ),
    ),
    _open: bool = Option(
        False,
        '-o',
        '--open',
        help='If true, open the index.html of the generated pages on completion',
    ),
):
    if not GH_PAGES.exists():
        _make_gh_pages()
    else:
        git_gh('pull')

    _clean()
    run_gh(('sphinx-apidoc', '-f', '-o', str(SOURCE_ROOT), str(CODE_ROOT)))
    run_gh(('sphinx-build', '-a', str(DOCS_ROOT), '.'))

    if commit_message:
        git_gh('add', '.')
        git_gh('commit', '-m', commit_message)
        git_gh('push')

    if _open:
        run_gh(('open', 'index.html'))


def _make_gh_pages():
    GH_PAGES.mkdir(parents=True)

    configs = git.configs()
    branches = git.branches()

    origin = configs['remote.origin.url']

    exists = 'gh-pages' in branches['origin']
    if exists:
        git_gh('clone', origin, '-b', 'gh-pages', '.')
    else:
        git_gh('clone', origin)

    git_gh('remote', 'add', 'upstream', UPSTREAM)
    git_gh('fetch', 'upstream')
    git_gh('switch', 'gh-pages')
    git_gh('reset', '--hard', 'upstream/gh-pages')
    git_gh('push', '--force-with-lease', '--set-upstream', 'origin', 'testing')


def _clean():
    if SOURCE_ROOT.exists():
        shutil.rmtree(SOURCE_ROOT)

    for i in GH_PAGES.iterdir():
        if i.name != '.git':
            if i.is_dir():
                shutil.rmtree(i)
            else:
                i.unlink()
