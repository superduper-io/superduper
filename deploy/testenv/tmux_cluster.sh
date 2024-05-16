#!/bin/bash

make testdb_init DB=mongodb

# Start a new tmux session
SESSION_NAME="debug_session"
tmux new-session -d -s $SESSION_NAME

# Split the window into panes. Adjust these commands to create the layout you prefer.
tmux split-window -h # Split the window horizontally
tmux split-window -v # Split the first pane vertically
tmux select-pane -t 0
tmux split-window -v # Split the first pane vertically again

# Set env variable to set configuration file
CFG=deploy/testenv/env/smoke/debug.yaml

# Send commands to each pane. Adjust these commands as per your requirement.
tmux send-keys -t $SESSION_NAME:0.0 "SUPERDUPERDB_CONFIG=$CFG PYTHONPATH=$(pwd):. ray start --head --dashboard-host=0.0.0.0 --disable-usage-stats --block" C-m 
tmux send-keys -t $SESSION_NAME:0.1 "SUPERDUPERDB_CONFIG=$CFG RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 PYTHONPATH=$(pwd):. ray start --address=localhost:6379  --block" C-m
tmux send-keys -t $SESSION_NAME:0.2 "SUPERDUPERDB_CONFIG=$CFG python -m superduperdb vector-search" C-m
tmux send-keys -t $SESSION_NAME:0.3 "SUPERDUPERDB_CONFIG=$CFG python -m superduperdb cdc" C-m 

# Attach to the session
tmux attach-session -t $SESSION_NAME
