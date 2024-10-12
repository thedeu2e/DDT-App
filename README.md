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
- DDT-MultiSwitch.py
- coco_socket_collector.py
- coco_socket_agent.py

## Commands

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
- Incorporate flow classification

## Suggested Reading

- **Syntax:** [ovs-ofctl.8](https://manpages.ubuntu.com/manpages/focal/en/man8/ovs-ofctl.8.html)
- **Match Fields:** [ovs-fields.7](https://manpages.ubuntu.com/manpages/focal/en/man7/ovs-fields.7.html)

## TODO
- Replace hard-coded file paths with relevant file paths for your machine.

## Running testbed
This research was conducted using a script based on [Mininet Flow Generator](https://github.com/stainleebakhla/mininet-flow-generator), please refer to repository. The following command will run this script:
