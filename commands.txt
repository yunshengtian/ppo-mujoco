# How to remote access Tensorboard running on server
https://gist.github.com/mrnabati/009c43b9b981ec2879008c7bff2fbb22

# tmux
# Reattach to session:
tmux attach -t <screen number>

# Kill session
ctrl+a+b -> : then type 'kill-session'

# Create new session with name
tmux new -s <name>

# Detach session
ctrl+a+b -> : then type 'detach'

# List tmux sessions
'tmux ls'

# Scroll in screen
`ctrl-a esc`

# Kill process
`sudo pkill -9 <PID>` in terminal
or
`sudo kill -9 <PID>` in terminal

# Kill all processes
`sudo killall -u dyung6`