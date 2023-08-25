# Quickstart

We use [`k6`](https://k6.io/) to load-test the 'Question the Docs' app. The deployment settings are controlled by the settings in `fly.toml`. A grafana dashboard for the deployment is also available at fly-metrics.net.

## To run the load-tests

1. Download the pre-built k6 binary with dashboard extension from [this](https://github.com/grafana/xk6-dashboard/releases/) page.
2. Move the k6 binary inside the current subdirectory (`stress/`)
3. (Optional) Tweak the number of virtual users (`vus`) inside `qtd-stress.js` to control the number of simulated users at the same time.
4. Run the shell script `./run-stress.sh`
5. After the shell script has finished, a window should open in the browser to view the results.