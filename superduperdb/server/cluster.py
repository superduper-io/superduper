import subprocess
import time


def subprocess_run(cmds):
    print(' '.join(cmds))
    subprocess.run(cmds)


def create_tmux_session(session_name, commands):
    '''
    Create a tmux local cluster
    '''

    window_name = f'{session_name}:0'
    for ix, cmd in enumerate(commands, start=1):
        if ix == 1:
            subprocess_run(
                [
                    'tmux',
                    'new-session',
                    '-d',
                    '-s',
                    session_name,
                    '-n',
                    'shell0',
                    '-d',
                    cmd,
                ]
            )
        else:
            subprocess_run(['tmux', 'split-window', '-t', window_name, cmd])

        if ix % 4 == 0:
            subprocess_run(['tmux', 'select-layout', '-t', window_name, 'tiled'])

        time.sleep(2)
    subprocess_run(['tmux', 'attach-session', '-t', session_name])


def local_cluster():
    print('Starting the local cluster...')
    session_name = 'superduperdb-localcluster-session'

    CFG = 'deploy/testenv/env/smoke/debug.yaml'
    services = [
        f"SUPERDUPERDB_CONFIG={CFG} PYTHONPATH=$(pwd):. "
        "ray start --head --dashboard-host=0.0.0.0 --disable-usage-stats --block",
        f"SUPERDUPERDB_CONFIG={CFG} RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 "
        "PYTHONPATH=$(pwd):. ray start --address=localhost:6379  --block",
        f"SUPERDUPERDB_CONFIG={CFG} python -m superduperdb vector-search",
        f"SUPERDUPERDB_CONFIG={CFG} python -m superduperdb cdc",
    ]

    create_tmux_session(session_name, services)

    print(f'local cluster started with tmux session {session_name}')
