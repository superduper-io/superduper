#!/bin/bash

# Start a new tmux session
SESSION_NAME="my_session"
tmux new-session -d -s $SESSION_NAME

# Split the window into panes. Adjust these commands to create the layout you prefer.
tmux split-window -h # Split the window horizontally
tmux split-window -v # Split the first pane vertically
tmux select-pane -t 0
tmux split-window -v # Split the first pane vertically again

# Send commands to each pane. Adjust these commands as per your requirement.
tmux send-keys -t $SESSION_NAME:0.0 'PYTHONPATH=./:.  ray start --head --dashboard-host=0.0.0.0 --disable-usage-stats --num-cpus=0 --block' C-m # Pane 1
tmux send-keys -t $SESSION_NAME:0.1 'RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 PYTHONPATH=./:. ray start --address=ray-head:6379  --block' C-m # Pane 1
tmux send-keys -t $SESSION_NAME:0.2 'SUPERDUPERDB_CLUSTER_COMPUTE=ray://localhost:10001 SUPERDUPERDB_ARTIFACT_STORE=filesystem:///tmp/artifacts SUPERDUPERDB_DATA_BACKEND=mongodb://superduper:superduper@mongodb:27017/test_db SUPERDUPERDB_CLUSTER_CDC_URI=http://cdc:8001 SUPERDUPERDB_CLUSTER_VECTOR_SEARCH=http://vector-search:8000 python -m superduperdb vector-search' C-m # Pane 2
tmux send-keys -t $SESSION_NAME:0.3 'SUPERDUPERDB_CLUSTER_COMPUTE=ray://localhost:10001 SUPERDUPERDB_ARTIFACT_STORE=filesystem:///tmp/artifacts SUPERDUPERDB_DATA_BACKEND=mongodb://superduper:superduper@mongodb:27017/test_db SUPERDUPERDB_CLUSTER_CDC_URI=http://cdc:8001 SUPERDUPERDB_CLUSTER_VECTOR_SEARCH=http://vector-search:8000 python -m superduperdb cdc' C-m # Pane 2

# Attach to the session
tmux attach-session -t $SESSION_NAME
