# README

The code for this repository was developed and debugged within a virtual environment using PyCharm Community Edition. It should be noted that there may be a moderate amount of troubleshooting to run files in this repository.

## Software
- **OS:** Ubuntu 24.04
- **Python:** 3.9 (Ryu does not support higher versions at the time of this)
- **Downgrade Dependencies:**
  - setuptools to 58.2.0
  - gunicorn to 20.1.0
  - eventlet to 0.30.2

## Install
- Mininet
- RYU SDN Controller
- MGEN traffic generator
- Numpy
- Pandas
- Torch

## Files
- Discrete SAC
- DDT-MultiSwitch.py
- coco_socket_collector.py
- coco_socket_agent.py
- prams.conf
- testbedM.py
- prams.conf

## Commands

### Run

```sh
sudo -E /home/user/pathtomyproject/venv/bin/python3 /home/user/pathtomyproject/app.py
sudo -E /home/thedeu2e/PycharmProjects/SDN-research/.venv/bin/python3.9 ./testbedM.py --controller=remote --topology=singular,5
sudo -E /home/thedeu2e/PycharmProjects/SDN-research/.venv/bin/python3.9 ./testbedM.py --controller=remote --topology=linear,5
sudo -E /home/thedeu2e/PycharmProjects/SDN-research/.venv/bin/python3.9 ./testbedM.py --controller=remote --topology=mesh,5
```

### Checks
```sh
ps aux | grep ryu-manager
ps aux | grep coco_socket_collector.py
ps aux | grep coco_socket_agent.py
```

### Terminates
```sh
pkill -f coco_socket_agent.py
pkill -f coco_socket_socket_collector.py
pkill -f ryu-manager
```

## Future Work
- Incorporate Adaptive “Low-Cost Near Real-Time Counters in SDN” code to forecast stats
- Consider adding additional data points (counters)
- Test more complex topologies (Mesh, Fat Tree)

## Suggested Reading

- **Syntax:** [ovs-ofctl.8](https://manpages.ubuntu.com/manpages/focal/en/man8/ovs-ofctl.8.html)
- **Match Fields:** [ovs-fields.7](https://manpages.ubuntu.com/manpages/focal/en/man7/ovs-fields.7.html)

## TODO
- Replace hard-coded file paths with relevant file paths for your machine.

## Running testbedM
This script is based on [Mininet Flow Generator](https://github.com/stainleebakhla/mininet-flow-generator), please refer to repository. The following command will run this script:

```sh
sudo -E /home/user/pathtomyproject/venv/bin/python3 /home/user/pathtomyproject/app.py
```

If the script is already made executable then we can simply run the script as:
```sh
./testbedM.py
```

### Command Line Arguments
- `--controller=remote[,ip=<ip_address>]`: This argument starts Mininet with a remote controller for our SDN network. The location of the controller can be specified with `ip_address`. If no IP address is specified, then it is assumed that the controller is running locally, and the value `localhost` is assumed. If this argument is not supplied, then Mininet starts its own default controller instead.
- `--topo=<topo_name>[,nodes]`: This argument specifies the topology with which Mininet should start. Available values are `linear`, `mesh`, and `fat_tree` topologies. The nodes specify the number of nodes to be spawned in that particular topology. If not specified, it assumes the value of 2. If this argument is not specified, then it assumes the value of linear topology with 2 nodes.
- `--debug`: If specified, this argument starts the python script in debug mode. It tries to connect to a locally available PyCharm IDE. If not specified, the script starts normally without opening any debug ports.

Once Mininet is started, it shows up the following prompt:

```sh
GEN/CLI/QUIT>_
```

### Commands

- **GEN**: This is the main purpose of the script. This command generates flows between two hosts randomly selected. This command takes 3 inputs: experiment time, number of elephant flows, and number of mice flows, and creates a schedule of those many elephant and mice flows selected at random and then executes them using `MGEN`. For each flow, two random hosts from the network are selected, one as a server and the other as a client. The server command is run at the host selected as the server, and the client command is run at the host selected as the client. This generates a traffic flow between two nodes in the network, which is picked up by ONOS for analysis. Once the flows have been generated, the script goes to sleep for the remaining duration of the experiment, to not disturb any of the ongoing flows. At the end of the experiment, the script starts up the cleaning process by removing all `MGEN` processes it had started. If there are any existing flows, these are stopped and killed immediately. However, care has been taken such that all flows generated with this command terminate well before the duration of the experiment.
- **CLI**: This command starts the Mininet CLI. Here you can enter commands, which are run by Mininet.
- **QUIT**: The command exits the script, shutting down Mininet and all the components, like switches, and links that it had started.

sudo rm -rf model_episode_0.h5
