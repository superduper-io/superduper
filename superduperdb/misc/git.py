from . import dataclasses as dc, run
from pathlib import Path
import typing as t

# Copied from https://github.com/rec/multi/blob/main/multi/git.py

# Separating git format fields by `|` means they can be easily parsed:
LOG_FLAGS = '--pretty=format:%h|%cd|%s', '--date=format:%g/%m/%d'
LONG_LOG_FLAGS = '--pretty=format:%h|%cd|%s', '--date=format:%g/%m/%d %H:%M:%S'


@dc.dataclass
class Git:
    out: t.Callable = run.out

    def __call__(self, *a, **ka):
        return self.out(('git', *a), **ka).splitlines()

    def commit(self, msg: str, *files, **kwargs) -> None:
        if not (files and self.is_dirty(**kwargs)):
            return

        files = [Path(f) for f in files]
        if exist := [f for f in files if f.exists()]:
            self('add', *exist, **kwargs)

        self('commit', '-m', msg, *files, **kwargs)
        self('push', **kwargs)

    def commit_all(self, msg: str, **kwargs) -> None:
        lines = self('status', '--porcelain', **kwargs)
        files = [i.split()[-1] for i in lines]

        self.commit(msg, files, **kwargs)

    def commits(self, *args, long: bool = False, **kwargs) -> t.List[str]:
        flags = LONG_LOG_FLAGS if long else LOG_FLAGS
        return self('log', *flags, *args, **kwargs)

    def is_dirty(self, unknown: bool = False, **kwargs) -> bool:
        lines = self('status', '--porcelain', **kwargs)
        if unknown:
            return any(lines)

        return any(not i.startswith('??') for i in lines)

    def configs(self, globals: t.Optional[bool] = None, **kwargs) -> t.Dict[str, str]:
        if globals:
            args = ('--global',)
        elif globals is False:
            args = '--local'
        else:
            args = ()
        configs = self('config', '--list', *args, **kwargs)
        return dict(c.partition('=')[::2] for c in configs)

    def current_branch(self, **kwargs) -> str:
        return self('symbolic-ref', '-q', '--short', 'HEAD')

    def branches(self, **kwargs) -> t.Dict[str, t.List[str]]:
        branches: t.Dict[str, t.List[str]] = {}
        for line in self('branch', '-r', **kwargs):
            remote, _, branch = line.partition('/')
            branches.setdefault(remote.strip(), []).append(branch.strip())
        return branches


git = Git()
