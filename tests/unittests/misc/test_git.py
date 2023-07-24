from superduperdb.misc.git import git
import os
import pytest

IN_CI = 'CI' in os.environ
PUNCTUATION = '.,-_'
TEST_COMMITS = os.environ.get('TEST_SUPERDUPERDB_COMMITS', 't').lower().startswith('t')
COMMIT_MESSAGES_TO_TEST = 10


def _errors(msg):
    if len(msg) > 50:
        yield 'length greater than 50'
    if msg[0].islower():
        yield 'does not start with a capital letter'
    if ps := [p for p in PUNCTUATION if msg.endswith(p)]:
        yield f'ends with punctuation: "{ps[0]}"'


def test_config():
    config = git.configs()
    assert isinstance(config, dict)
    if IN_CI:
        assert 10 < len(config) < 20
    else:
        assert len(config) > 20
        assert 'user.email' in config


def test_commit_errors():
    msg = 'this is a very very very very very very very bad error message.'

    actual = ', '.join(_errors(msg))
    expected = (
        'length greater than 50, '
        'does not start with a capital letter, '
        'ends with punctuation: "."'
    )
    assert actual == expected


@pytest.mark.skipif(not TEST_COMMITS, reason='Not testing commit names')
def test_last_commit_messages():
    # If this test fails, it's because one of your last commit messages was suboptimal!
    #
    # If you want to skip this test, set an environment variable:
    #
    #   TEST_SUPERDUPERDB_COMMITS=false

    commits = git.commits(F'-{COMMIT_MESSAGES_TO_TEST + 1}')
    commits.pop(0 if IN_CI else -1)

    bad_commits = []

    for commit in commits:
        commit_id, date, msg = commit.split('|')
        if errors := ', '.join(_errors(msg)):
            bad_commits.append(f'{commit_id}: {msg}:\n    {errors}')

    if bad_commits:
        print('Bad commits:', *bad_commits, sep='\n')
        assert not bad_commits
