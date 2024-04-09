import subprocess
import time


def create_tmux_session(session_name, commands):
    '''
    Create a tmux local cluster
    '''

    for ix, cmd in enumerate(commands, start=0):
        window_name = f'{session_name}:0.{ix}'
        run_tmux_command(['send-keys', '-t', window_name, cmd, 'C-m'])
        time.sleep(1)

    run_tmux_command(['attach-session', '-t', session_name])


def run_tmux_command(command):
    print('tmux ' + ' '.join(command))
    subprocess.run(["tmux"] + command, check=True)


def local_cluster():
    print('Starting the local cluster...')
    session_name = 'superduperdb-localcluster-session'

    CFG = 'deploy/testenv/env/smoke/debug.yaml'
    run_tmux_command(
        [
            'new-session',
            '-d',
            '-s',
            session_name,
        ]
    )

    run_tmux_command(["split-window", "-h"])
    run_tmux_command(["split-window", "-v"])
    run_tmux_command(["select-pane", "-t", "0"])
    run_tmux_command(["split-window", "-v"])

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
