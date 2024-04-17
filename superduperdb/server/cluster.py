import os
import subprocess
import sys
import time

session_name = 'superduperdb-local-cluster-session'

def create_tmux_session(session_name, commands):
    '''
    Create a tmux local cluster
    '''

    for ix, cmd in enumerate(commands, start=0):
        window_name = f'{session_name}:0.{ix}'
        run_tmux_command(['send-keys', '-t', window_name, cmd, 'C-m'])
        time.sleep(1)

    print('You can attach to the tmux session with:')
    print(f'tmux attach-session -t {session_name}')


def run_tmux_command(command):
    print('tmux ' + ' '.join(command))
    subprocess.run(["tmux"] + command, check=True)


def up_cluster():
    print('Starting the local cluster...')

    CFG = os.environ.get('SUPERDUPERDB_CONFIG')
    assert CFG, 'Please set SUPERDUPERDB_CONFIG environment variable'
    python_executable = sys.executable
    ray_executable = os.path.join(os.path.dirname(python_executable), 'ray')
    run_tmux_command(
        [
            'new-session',
            '-d',
            '-s',
            session_name,
        ]
    )

    run_tmux_command(["split-window", "-h", "-t", f"{session_name}:0.0"])
    run_tmux_command(["split-window", "-v", "-t", f"{session_name}:0.0"])
    run_tmux_command(["split-window", "-v", "-t", f"{session_name}:0.0"])

    services = [
        f"SUPERDUPERDB_CONFIG={CFG} PYTHONPATH=$(pwd):. "
        f"{ray_executable} start --head --dashboard-host=0.0.0.0 --disable-usage-stats --block",
        f"SUPERDUPERDB_CONFIG={CFG} RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 "
        f"PYTHONPATH=$(pwd):. {ray_executable} start --address=localhost:6379  --block",
        f"SUPERDUPERDB_CONFIG={CFG} {python_executable} -m superduperdb vector-search",
        f"SUPERDUPERDB_CONFIG={CFG} {python_executable} -m superduperdb cdc",
    ]

    create_tmux_session(session_name, services)

    print(f'local cluster started with tmux session {session_name}')

def down_cluster():
    print('Stopping the local cluster...')
    run_tmux_command(['kill-session', '-t', session_name])
    print('local cluster stopped')

def attach_cluster():
    run_tmux_command(['attach-session', '-t', session_name])
