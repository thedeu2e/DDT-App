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

from singular import SingularTopo
from linear import LinearTopo
#from mesh import MeshTopo
#from fat_tree_topo import FatTreeTopo

from os import path
from os import mkdir
import random
import time
import sys
import re
import numpy as np
#import pandas as pd
import subprocess
import json
import os
import signal

print(sys.executable)

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
    x = random.randint(1,10)
    if x == 1:
        y = 10
    else:
        y = (random.randint(round((10/x)+1), 10))
    i = str(round((1/x), 2))
    n = str(y)
    
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
    x = random.randint(1,10)
    y= (random.randint(1, round(10/x)))
    i = str(round((1/x), 2))
    n = str(y)
    
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

    # Add validation to ensure interval is at least 1
    if interval < 1:
        interval = 1

    # Setting random mice flow or elephant flows
    flow_type = ['E'] * n_elephant_flows + ['M'] * n_mice_flows
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
    # killing the MGEN sessions
    info("Stopping traffic...\n")
    info("Killing active MGEN sessions...\n")

    # killing iperf in all the hosts
    #for host in net.hosts:
        #host.cmdPrint('killall -9 mgen')
    """

def handle_interrupt(signum, frame):
    print("Received KeyboardInterrupt. Stopping Ryu controller application...")
    # Kill subprocesses
    subprocess.run("pkill -f coco_socket_agent.py", shell=True, check=False)
    subprocess.run("pkill -f coco_socket_collector.py", shell=True, check=False)
    subprocess.run("pkill -f ryu-manager", shell=True, check=False)
    # Add any cleanup or graceful exit logic here
    net.stop()
    sys.exit()

# Register the signal handler for KeyboardInterrupt
signal.signal(signal.SIGINT, handle_interrupt)

class Configuration:

    def readConfig(self, filename):
        file = open(filename, "r")
        if filename:
            with open(filename) as f:
                data = json.load(f)

        return dict(data)

    def writeConfig(self, filename, output_filename, eta, T0, beta_init, beta, W):
        file = open(filename, 'w')

        file.write('[DEFAULT]\n\n')

        file.write('eta = {}\n'.format(eta))
        file.write('T0 = {}\n'.format(T0))
        file.write('beta_init = {}\n'.format(beta_init))
        file.write('beta = {}\n'.format(beta))
        file.write('W = {}\n'.format(W))
        file.write('output_filename = {}\n'.format(output_filename))

        file.close()
        

# Main function
if __name__ == "__main__":
    # Initialize the controller
    ryu_controller = '/home/thedeu2e/PycharmProjects/SDN-research/.venv/bin/ryu-manager /home/thedeu2e/PycharmProjects/SDN-research/DDT-MultiSwitch.py &'
    print("Controller Command:", ryu_controller)
    try:
        stdout_file = open("/home/thedeu2e/PycharmProjects/SDN-research/output.txt", "w")
        errout_file = open("/home/thedeu2e/PycharmProjects/SDN-research/errout.txt", "w")
        pid_controller = subprocess.Popen(ryu_controller, shell=True, stdout=stdout_file, stderr=errout_file, encoding='utf-8')
    except Exception as e:
        print(f"Error starting ryu controller subprocess: {e}")

    # Allow the controller time to initialize
    time.sleep(3)
    
    # Loading default parameter values
    log_dir = "/Desktop/mininet-log/test-"
    topology = SingularTopo()
    default_controller = True
    controller_ip = '127.0.0.1'  # localhost
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
            if sub_arg[1] == "singular":
                if len(sub_arg) == 3:
                    n = int(sub_arg[2])
                    topology = SingularTopo()
            if sub_arg[1] == "linear":
                if len(sub_arg) == 3:
                    n = int(sub_arg[2])
                    topology = LinearTopo(n)
            """
            if sub_arg[1] == "mesh":
                if len(sub_arg) == 3:
                    n = int(sub_arg[2])
                    topology = MeshTopo(n)
                else:
                    topology = MeshTopo()
            elif sub_arg[1] == "fat_tree":
                topology = FatTreeTopo()
            """

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
    # allow collector time to initialize
    #time.sleep(45)
    
    # starting mininet
    if default_controller:
        net = Mininet(topo=topology, controller=DefaultController, host=CPULimitedHost, link=TCLink,
                      switch=OVSSwitch, autoSetMacs=True)
    else:
        net = Mininet(topo=topology, controller=None, host=CPULimitedHost, link=TCLink,
                      switch=OVSSwitch, autoSetMacs=True)
        net.addController('c1', controller=RemoteController, ip=controller_ip, port=controller_port)

    net.start()
    
    fixed_filename = 'fixed.csv'

    params_filename = '/home/thedeu2e/PycharmProjects/SDN-research/params.conf'
    config = Configuration()

    beta_init_tab = [50]  # i
    beta_tab = [10]  # j      #W = int(1.5*beta_init)
    eta_tab = [0]  # [0.1] => to use COCO  #k MASE
    T0_tab = [0.5]  # t

    dataset_id = 0

    num_simu = 0
    #err_pd = pd.DataFrame(columns=['num_simu', 'eta', 'T0', 'beta_init', 'beta', 'W'])
    #global_pd = pd.DataFrame(columns=['output_filename', 'eta', 'T0', 'beta_init', 'beta', 'W'])

    rep = "/home/thedeu2e/PycharmProjects/SDN-research/Outputs/{0}/".format(dataset_id)
    global_filename = '{0}_global.csv'.format(dataset_id)
    global_err_file_filename = '{0}_global_err.csv'.format(dataset_id)
    fixed_filename = ''

    TIME_DEBUT = time.time()
    
    for k in range(len(eta_tab)):
        eta = eta_tab[k]
        if (eta == 0):
            for t in range(len(T0_tab)):
                T0 = T0_tab[t]
                file_id = (time.strftime("%c")).replace(" ", "_")  # actual date and time
                output_filename = rep + "COCO_CTRL_" + file_id + "_.csv"
                fixed_filename = output_filename[:len(output_filename) - 4] + 'fixed_push.csv'
                print(f"Creating file: {fixed_filename}")
                config.writeConfig(params_filename, output_filename, eta, T0, 0, 0, 0)
                #new_pd = pd.DataFrame([[output_filename, eta, T0, 0, 0, 0]],
                                      #columns=['output_filename', 'eta', 'T0', 'beta_init', 'beta', 'W'])
                #global_pd = pd.concat([global_pd, new_pd])
                #global_pd.to_csv(global_filename)
                num_simu = num_simu + 1
        else:
            T0 = T0_tab[1]  # T0 = 1
            for i in range(len(beta_init_tab)):
                beta_init = beta_init_tab[i]
                W = int(1.5*beta_init)
                for j in range(len(beta_tab)):
                    beta = beta_tab[j]
                    file_id = (time.strftime("%c")).replace(" ", "_")      # actual date and time
                    output_filename = rep + "COCO_CTRL_" + file_id +"_.csv"
                    fixed_filename = output_filename[:len(output_filename)-4] + 'fixed_push.csv'
                    print(f"Creating file: {fixed_filename}")
                    config.writeConfig(params_filename, output_filename, eta, T0, beta_init, beta, W)
                    #new_pd = pd.DataFrame([[output_filename, eta, T0, beta_init, beta, W]], columns=['output_filename', 'eta', 'T0', 'beta_init', 'beta', 'W'])
                    #global_pd = pd.concat([global_pd, new_pd])
                    #global_pd.to_csv(global_filename)
                    num_simu = num_simu + 1
                
    # Iterate through switches
    for switch in net.switches:
        agent_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coco_socket_agent.py')
        try:
            # switch_fixed_filename = f'{fixed_filename}_{switch.name}'
            # print(switch_fixed_filename)
            subprocess.run('python3 {} --switch-name {} --fixed-filename {} &'.format(agent_script, switch.name, fixed_filename), shell=True, check=True)
            info("Socket File Called by {}...\n".format(switch.name))
        except subprocess.CalledProcessError as e:
            print(f"Error calling script for {switch.name}: {e}")
            
    user_input = "GEN"

    # run till user quits
    while True:
        if user_input.upper() == "GEN":
            experiment_duration = 1200500
            n_elephant_flows = 240000
            n_mice_flows = 960000

            generate_flows(n_elephant_flows, n_mice_flows, experiment_duration, net)

        elif user_input.upper() == "CLI":
            info("Running CLI...\n")
            CLI(net)

        elif user_input.upper() == "QUIT":
            info("Terminating...\n")
            handle_interrupt(signal.SIGINT, None)  # Trigger the graceful exit logic
            break

        else:
            print("Command not found")

    # if user enters CTRL + D then treat it as quit
        try:
            user_input = input("GEN/CLI/QUIT: ")
        except EOFError as error:
            user_input = "QUIT"

'''
Area for scratch pad

600,500 duration
60,000 elephant flows
540,000 mice flows

os.path() + '/.venv/bin/ryu-manager
'''
