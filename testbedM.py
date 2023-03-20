#!/usr/bin/python
"""
Custom topology launcher Mininet, with traffic generation using MGEN
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, DefaultController
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

from linear import LinearTopo
from mesh import MeshTopo
from fat_tree_topo import FatTreeTopo

from os import path
from os import mkdir
import random
import time
import sys
import re
import numpy as np

mice_flow_min = 100  # KBytes = 100KB
mice_flow_max = 10240  # KBytes = 10MB
elephant_flow_min = 10240  # KBytes = 10MB
elephant_flow_max = 1024*1024*10  # KBytes = 10 GB

# MICE FLOW PARAMS
mice_bandwidth_list = [128, 256, 512, 1024]

# ELEPHANT FLOW PARAMS
elephant_bandwidth_list = [1024, 2048, 4068, 8192]

# L4 PROTOCOLS
protocol_list = ["UDP", "TCP"]  # udp / tcp
port_min = 1024
port_max = 65535

# CMD PARAMS
intervals = [1, 0.5, 0.3, 0.25, 0.2, 0.16, 0.14, 0.125, 0.11, 0.10]
number = [2, 3, 4, 5, 6, 7, 8, 9, 10] 


def random_normal_number(low, high):
    range = high - low
    mean = int(float(range) * float(75) / float(100)) + low
    sd = int(float(range) / float(4))
    num = np.random.normal(mean, sd)
    return int(num)


def generate_elephant_flows(id, net):

    """
    Generate Elephant flows
    May use either tcp or udp
    """
    
    hosts = net.hosts

    # select random src and dst
    end_points = random.sample(hosts, 2)
    src = net.get(str(end_points[0]))
    dst = net.get(str(end_points[1]))
    
    # select connection params
    protocol = random.choice(protocol_list)
    port_argument = str(random.randint(port_min, port_max))
    bandwidth_argument = random.choice(elephant_bandwidth_list)
    i = str(random.choice(intervals))
    n = str(random.choice(number))
    
    # create cmd
    client_cmd = "mgen event \"ON "
    client_cmd += str(id) + " "
    client_cmd += protocol
    client_cmd += " DST "
    client_cmd += dst.IP() + "/"
    client_cmd += port_argument
    client_cmd += " PERIODIC ["
    client_cmd += i + " "
    client_cmd += str(bandwidth_argument)
    client_cmd += "] count "
    client_cmd += n
    client_cmd += ",off\" &"


    # send the cmd
    src.cmdPrint(client_cmd)


def generate_mice_flows(id, net):

    """
    Generate mice flows
    May use either tcp or udp
    """
    
    hosts = net.hosts

    # select random src and dst
    end_points = random.sample(hosts, 2)
    src = net.get(str(end_points[0]))
    dst = net.get(str(end_points[1]))
    
    # select connection params
    protocol = random.choice(protocol_list)
    port_argument = str(random.randint(port_min, port_max))
    bandwidth_argument = random.choice(mice_bandwidth_list)
    i = str(1)
    n = str(1)
    
    # create cmd
    client_cmd = "mgen event \"ON "
    client_cmd += str(id) + " "
    client_cmd += protocol
    client_cmd += " DST "
    client_cmd += dst.IP() + "/"
    client_cmd += port_argument
    client_cmd += " PERIODIC ["
    client_cmd += i + " "
    client_cmd += str(bandwidth_argument)
    client_cmd += "] count "
    client_cmd += n
    client_cmd += ",off\" &"


    # send the cmd
    src.cmdPrint(client_cmd)


def generate_flows(n_elephant_flows, n_mice_flows, duration, net):
    """
    Generate elephant and mice flows randomly for the given duration
    """

    """""
    if not path.exists(log_dir):
        mkdir(log_dir)
    """""

    n_total_flows = n_elephant_flows + n_mice_flows
    interval = duration / n_total_flows

    # setting random mice flow or elephant flows
    flow_type = []
    for i in range(n_elephant_flows):
        flow_type.append('E')
    for i in range(n_mice_flows):
        flow_type.append('M')
    random.shuffle(flow_type)

    # setting random flow start times
    flow_start_time = []
    for i in range(n_total_flows):
        n = random.randint(1, int(interval))
        if i == 0:
            flow_start_time.append(0)
        else:
            flow_start_time.append(flow_start_time[i - 1] + n)

    """
    # setting random flow end times
    # using normal distribution
    # we will keep duration till 95% of the total duration
    # the remaining 5% will be used as buffer to finish off the existing flows
    flow_end_time = []
    for i in range(n_total_flows):
        s = flow_start_time[i]
        print(s)
        e = int(float(95) / float(100) * float(duration))  # 95% of the duration
        print(e)
        end_time = random_normal_number(s, e)
        while end_time > e:
            end_time = random_normal_number(s, e)
        flow_end_time.append(end_time)

    # calculating flow duration from start time and end time generated above
    flow_duration = []
    for i in range(n_total_flows):
        flow_duration.append(flow_end_time[i] - flow_start_time[i])
    """
    print(flow_type)
    print(flow_start_time)
    
    """
    print(flow_end_time)
    print(flow_duration)
    print("Remaining duration :" + str(duration - flow_start_time[-1]))
    """

    # generating the flows
    for i in range(n_total_flows):
        time.sleep(1)
        for j in range(10):
            if flow_type[i] == 'E':
                generate_elephant_flows(i, net)
            elif flow_type[i] == 'M':
                generate_mice_flows(i, net)

    # sleeping for the remaining duration of the experiment
    remaining_duration = duration - flow_start_time[-1]
    info("Traffic started, going to sleep for %s seconds...\n " % remaining_duration)
    # time.sleep(remaining_duration)

    """
    # ending all the flows generated by
    # killing the iperf sessions
    info("Stopping traffic...\n")
    info("Killing active iperf sessions...\n")

    # killing iperf in all the hosts
    #for host in net.hosts:
        #host.cmdPrint('killall -9 iperf')
    """

# Main function
if __name__ == "__main__":
    # Loading default parameter values
    log_dir = "/Desktop/mininet-log/test-"
    topology = LinearTopo()
    default_controller = True
    controller_ip = "127.0.0.1"  # localhost
    controller_port = 6633
    debug_flag = False
    debug_host = "localhost"
    debug_port = 6000

    # Reading command line arguments
    for arg in sys.argv:
        if arg.startswith("--controller"):
            default_controller = False
            arg = arg[2:]
            sub_arg = re.split("[,=]", arg)
            if "ip" in sub_arg:
                index = sub_arg.index("ip") + 1
                controller_ip = sub_arg[index]
            if "port" in sub_arg:
                index = sub_arg.index("port") + 1
                controller_port = int(sub_arg[index])

        elif arg.startswith("--topo"):
            arg = arg[2:]
            sub_arg = re.split("[,=]", arg)
            if sub_arg[1] == "linear":
                if len(sub_arg) == 3:
                    n = int(sub_arg[2])
                    topology = LinearTopo(n)
            if sub_arg[1] == "mesh":
                if len(sub_arg) == 3:
                    n = int(sub_arg[2])
                    topology = MeshTopo(n)
                else:
                    topology = MeshTopo()
            elif sub_arg[1] == "fat_tree":
                topology = FatTreeTopo()

        elif arg.startswith("--debug"):
            debug_flag = True
            if len(arg) > 7:
                arg = arg[8:]
                sub_arg = re.split(":", arg)
                debug_host = sub_arg[0]
                debug_port = int(sub_arg[1])

            sys.path.append("/home/stainlee/Programs/pycharm-2017.3.3/debug-eggs/pycharm-debug.egg")
            import pydevd

            # conecting to pycharm debugger
            pydevd.settrace(debug_host, port=debug_port, stdoutToServer=True, stderrToServer=True)

    # Starting program
    setLogLevel('info')
    """""
    # creating log directory
    log_dir = path.expanduser('~') + log_dir
    i = 1
    while True:
        if not path.exists(log_dir + str(i)):
            # mkdir(log_dir + str(i))
            log_dir = log_dir + str(i)
            break
        i = i+1
    """""
    # starting mininet
    if default_controller:
        net = Mininet(topo=topology, controller=DefaultController, host=CPULimitedHost, link=TCLink,
                      switch=OVSSwitch, autoSetMacs=True)
    else:
        net = Mininet(topo=topology, controller=None, host=CPULimitedHost, link=TCLink,
                      switch=OVSSwitch, autoSetMacs=True)
        net.addController('c1', controller=RemoteController, ip=controller_ip, port=controller_port)

    net.start()

    user_input = "QUIT"

    # run till user quits
    while True:
        # if user enters CTRL + D then treat it as quit
        try:
            user_input = input("GEN/CLI/QUIT: ")
        except EOFError as error:
            user_input = "QUIT"

        if user_input.upper() == "GEN":
            experiment_duration = int(input("Experiment duration: "))
            n_elephant_flows = int(input("No of elephant flows: "))
            n_mice_flows = int(input("No of mice flows: "))

            generate_flows(n_elephant_flows, n_mice_flows, experiment_duration, net)

        elif user_input.upper() == "CLI":
            info("Running CLI...\n")
            CLI(net)

        elif user_input.upper() == "QUIT":
            info("Terminating...\n")
            net.stop()
            break

        else:
            print("Command not found")

'''
Area for scratch pad

'''
