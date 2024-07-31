# -----------------------------------------------------------
# (C) 2024 Nathan Harris, Jr., Greensboro, North Carolina
# Released under the MIT License (MIT)
# email ncharris1@aggies.ncat.edu
# -----------------------------------------------------------

# required for misc. aspects of program
from datetime import datetime

# required for collector socket processes
import numpy as np

# required for cache
from cachetools import TTLCache

# base classes/libraries
from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
# required for Layer 4 matching
from ryu.lib.packet import in_proto
from ryu.lib.packet import ipv4
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.lib.packet import tcp
from ryu.lib.packet import udp

# required for collector
import pickle
import time
import sys
import math
import socket
from _thread import *

#  Soft Actor-Critic for
from DiscreteSAC import Discrete_SAC_Agent
import torch

#  visualize rewards for analysis
import matplotlib.pyplot as plt

# RL Model Parameters & Constants (these can be defined outside this method or within your class)
STATE_DIM = 5  # 4-Dimensional State Space: [avg_PI_IAT, avg_fd, Inactive, action, misses]
ACTION_DIM = 10  # 10-Dimensional Action Space: 1-10
MAX_EPISODES = 2000  # the maximum number of episodes used to train the model (4000 seconds in episodes * 1000
# episodes = 300,000 duration / 5 polling periods = 60,000 training steps)
MAX_EPISODE_STEPS = 30  # the maximum number of steps per episode (60 seconds/20 second increments send polling
# intervals)
TRAINING_EVALUATION_RATIO = 4  # How often to evaluate
evaluation_episode = False
SAVE_INTERVAL = 50

poll = 5  # polling increments in seconds

#  collector parameters for a socket connection
HOST = "127.0.0.1"  # switch addr, the IP address (localhost) where the socket server is running or where the client
# will connect
PORT = 8888  # port number on which the socket server is listening or where the client will connect.
collected_data = []  # empty lists that seem intended to store collected data
deviations_data = []  # empty lists that seem intended to store deviations
#  a list of strings containing field names such as 'n_packets', 'n_bytes', and 'duration'
ts_fields_ = ['n_packets', 'n_bytes', 'duration']
# y::value = function('n_packets', 'n_bytes', 'duration'), with the addition of 'time' at the beginning
# and 'value' at the end. This represents the fields that will be included in a time series data structure
ts_fields = ['time'] + ts_fields_ + ['value']

# Monitoring metrics computation: network utilization
total_BW = 10000000  # 10 Mbps


def ts_function1(last_collected_n_bytes, actual_collected_n_bytes,
                 talpha):  # last_collected_n_bytes: The number of bytes collected in the last measurement.
    # actual_collected_n_bytes: The current number of bytes collected in the latest measurement. talpha: The time
    # difference between the last and current measurements.
    # ts value y, monitoring time series stream depending on several flow stats
    # network utilization
    utilization = ((actual_collected_n_bytes - last_collected_n_bytes) * 8) / (talpha)  # in %
    return utilization  # return network utilization, measured in percentage


# This function has no parameters and always returns zero
def ts_function():
    utilization = 0
    return utilization

class SimpleMonitor13(simple_switch_13.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}  # dictionary of switches in network

        self.monitor_thread = hub.spawn(self._monitor)
        self.switches = {}  # a dictionary of switches within the network to keep track of key:flow rule entries,
        # value:packet count pairs
        self.avg_PI_IAT = 0  # average packet in message inter-arrival time of misses | feature (state)
        self.avg_fd = 0  # average flow duration from flow removed message | feature (state)
        self.curr_count = 0  # current number of flows in flow table from table stats reply
        self.fr_counter = 0  # running total of flows that have been removed from flow table from flow removed
        self.total_dur = 0  # running total of duration for flows removed from flow removed message
        self.hit = 0  # percentage of packets matched from table stats reply | outcome (reward) | feature (state)
        self.use = 0  # percentage of active flows from table stats reply | outcome (reward) | feature (state)
        self.action = 10  # timeout value | feature (state)
        self.counter = 0  # total number of packet in request(s)
        self.cache = TTLCache(maxsize=1000, ttl=20)  # cache where each item is accessible for 10s
        self.misses = 0  # flow table misses
        self.difference = 0  # sum of packet in inter-arrival time difference
        self.total_pi = 0  # total count of packet_in messages
        self.hitSum = 0  # sum of hits
        self.useSum = 0  # sum of active
        self.avg_use = 0  # average use rate
        self.avg_hit = 0  # average hit rate
        self.pi_count = 0  # total packet in count for episode
        self.miss_pi = 0  # sum of missed packet in messages
        self.holder1 = 0
        self.holder2 = 0
        self.switch_uses = []  # holds individual table statistics

        # RL Algorithm initialization specific 
        self.model = Discrete_SAC_Agent.SACAgent(STATE_DIM, ACTION_DIM)  # SACD initialization
        #  self.replay_buffer = DiscreteSAC.utilities.ReplayBuffer(STATE_DIM, ACTION_DIM) Replay Buffer initialization
        self.prev_state = np.array([None, None, None, None, None])  # placeholder for previous state
        self.state = np.array([None, None, None, None, None])  # placeholder for current state
        self.episode = 0  # episode counter initialization
        self.episode_step = 0  # episode step counter initialization
        self.miniep = 0  # single step increments
        self.episode_reward = 0  # collects reward value of evaluation episodes
        self.evaluation_rewards = []  # collect performance results of evaluation episodes

        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0001  # exponential decay rate for exploration prob

        # Evaluation
        self.tp = 0  # total count of packet_in messages
        self.avg_active = 0  # total avg of active flows
        self.avg_rate = 0  # total avg of table hits
        self.iterations = 0  # total times ran
        self.holder3 = 0
        self.holder4 = 0

        # ported socket vars
        self.TIME_DEBUT = time.time()
        self.switch_data = {}
        self.eta = 0
        self.T0 = 5
        self.beta_init = 0
        self.beta = 0
        self.W = 0
        self.coco_collector_socket = self.create_socket()

        start_new_thread(self.collection_initialization)

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
                self.switches[datapath.id] = {}

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]
                del self.switches[datapath.id]

    #  This method should bind our socket to a host ip address and a port
    def create_socket(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
        except socket.error:
            print('Failed to create socket')
            sys.exit()
        return s

    def collection_initialization(self):
        time_ = round((time.time() - self.TIME_DEBUT), 6)
        initialization_data = {"time": time_, "eta": self.eta, "t0": self.T0, "beta_init": self.beta_init,
                               "beta": self.beta}
        msg = pickle.dumps(initialization_data)
        print("BEN =====================> CTRL expresses its interest ...")

        # Print socket information
        local_addr = self.coco_collector_socket.getsockname()
        print(f"Server socket bound to: {local_addr}")

        self.coco_collector_socket.listen()
        # Wait for connections
        # Spawn a new thread each time a connection is received to handle the response and ongoing communication
        try:
            while True:
                c, addr = self.coco_collector_socket.accept()

                print('Connected to:', addr[0], ':', addr[1])

                start_new_thread(self.connection_handler, (c, addr[0], msg))
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught. Closing gracefully...")
            self.cleanup()

    def connection_handler(self, connection, ipaddress, msg):
        print('Successful connection from ' + ipaddress)

        try:
            connection.sendall(msg)  # Tries to send the serialized data to a specified host and port

            # Wait for confirmation
            switch = connection.recv(1024).decode('utf-8')

            if switch not in self.switch_data:
                self.switch_data[switch] = []

            self.fixed_collect(connection, switch)

        except KeyboardInterrupt:
            print("KeyboardInterrupt at Initialization. Closing gracefully...")
            self.cleanup()

    def cleanup(self):
        # Print socket information before closing
        local_addr = self.coco_collector_socket.getsockname()
        print(f"Closing server socket bound to: {local_addr}")

        # Add cleanup code here, such as closing sockets, releasing resources, etc.
        if hasattr(self, 'coco_collector_socket'):
            self.coco_collector_socket.close()
        print("Cleaning up and exiting gracefully...")
        sys.exit()

    def fixed_collect(self, connection, switch):
        print("BEN==========>Collector fixed Collection")
        try:
            while True:
                data = connection.recv(1024)
                # sample = data.decode().strip()
                sample = pickle.loads(data)
                if sample:
                    try:
                        sample = float(sample)
                        self.switch_data[switch] = sample
                        print('>>>>>> collected data == ', sample)
                    except ValueError:
                        # Handle the case where the string is not a valid float
                        print("Invalid float representation:", repr(sample))
                else:
                    # Handle the case where the received data is empty
                    print("Received empty data")
        except KeyboardInterrupt:
            print("KeyboardInterrupt during data collection. Closing gracefully...")
            self.cleanup()

    def _monitor(self):
        self.logger.info("starting flow monitoring thread")

        #self.model.load('/home/thedeu2e/PycharmProjects/SDN-research/model')
        #self.logger.info("loaded")

        while True:

            while self.episode < MAX_EPISODES:
                evaluation_episode = self.episode % TRAINING_EVALUATION_RATIO == 0
                # reset episode variables
                self.episode_step = 0
                self.fr_counter = 0
                self.total_dur = 0
                self.difference = 0
                self.miss_pi = 0
                self.episode_reward = 0

                while self.episode_step < MAX_EPISODE_STEPS:
                    # reset episode step variables
                    self.miniep = 0
                    self.hitSum = 0
                    self.useSum = 0
                    self.holder1 = 0
                    self.holder2 = 0

                    if self.episode_step == 0:
                        # initialize state and previous state array
                        self.state = np.array([0, 10, 0, float(self.action), 0], dtype=float)
                        self.prev_state = np.array([0, 0, 0, float(self.action), 0])
                    else:
                        # Reset the state each time
                        self.state = np.array(
                            [self.prev_state[0], self.prev_state[1], 0, self.action, self.prev_state[4]],
                            dtype=float)

                    while self.miniep < 4:
                        # Perform actions within this loop
                        # displays current episode, episode step, and ministep
                        self.logger.info("Episode: %s Step: %s Mini-Step: %s", self.episode, self.episode_step,
                                         self.miniep)

                        self.dynamic_timeout()

                        # displays current timeout value
                        self.logger.info("timeout value: %s", self.action)

                        # increment miniep counter
                        self.miniep += 1

                        # thread sleeps for new duration selected by agent
                        hub.sleep(poll)

                    # displays current state of network
                    self.logger.info("Current State:%s ", self.state)


                    #  increment episode step counter after processing miniep transitions
                    self.episode_step += 1

                if evaluation_episode:
                    self.evaluation_rewards.append(self.episode_reward)
                    #self.save_evaluation_reward(self.episode_reward)

                # Save the model at regular intervals
                if self.episode % SAVE_INTERVAL == 0:
                    save_path = f'/home/thedeu2e/PycharmProjects/SDN-research/model_episode_{self.episode}.h5'
                    self.model.save(save_path)
                    self.logger.info("Intermediate save at episode %s to %s", self.episode, save_path)

                # increment episode counter
                self.episode += 1
            break

        # Final save after completing all episodes
        final_save_path = '/home/thedeu2e/PycharmProjects/SDN-research/model_final.h5'
        self.model.save(final_save_path)
        self.logger.info("Final save after all episodes completed to %s", final_save_path)

        #self.plot_results()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install the table-miss flow entry.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, **kwargs):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # construct flow_mod message and send it.
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                command=ofproto.OFPFC_ADD, idle_timeout=self.action,
                                flags=ofproto.OFPFF_SEND_FLOW_REM,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn,
                MAIN_DISPATCHER)  # Using 'MAIN_DISPATCHER' as the second argument means this function is called only after the negotiation completes
    def _packet_in_handler(self, ev):
        self.total_pi += 1  # sum of packet_in messages during polling period
        self.tp += 1  # sum of packet_in messages during testing

        msg = ev.msg  # object that represents a packet_in data structure
        datapath = msg.datapath  # an object that represents a datapath (switch)
        ofproto = datapath.ofproto  # an object that represent the OpenFlow protocol that Ryu and the switch negotiated
        parser = datapath.ofproto_parser  # object that represents the OpenFlow protocol that Ryu & the switch negotiated
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore Link Layer Discovery Protocol packet
            return
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:

            # check IP Protocol and create a match for IP
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                srcip = ip.src
                dstip = ip.dst
                protocol = ip.proto

                # if ICMP Protocol
                if protocol == in_proto.IPPROTO_ICMP:
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=srcip,
                                            ipv4_dst=dstip,
                                            ip_proto=protocol)

                #  if TCP Protocol
                elif protocol == in_proto.IPPROTO_TCP:
                    t = pkt.get_protocol(tcp.tcp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, eth_dst=dst, eth_src=src,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol, )  #tcp_src=t.src_port, tcp_dst=t.dst_port, )

                #  If UDP Protocol
                elif protocol == in_proto.IPPROTO_UDP:
                    u = pkt.get_protocol(udp.udp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, eth_dst=dst, eth_src=src,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol, )  #udp_src=u.src_port, udp_dst=u.dst_port, )

                # verify if we have a valid buffer_id, if yes avoid to send both
                # flow_mod & packet_out
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions)

                flow = str(match)  # create key

                if flow not in self.cache:  # search for key in cache
                    self.cache[flow] = datetime.now()  # if the flow isn't in the cache, add it with time it was added
                else:
                    now = datetime.now()  # current time
                    self.difference += (now - self.cache[flow]).seconds  # add difference of times to sum
                    self.cache[flow] = now  # update flow's time
                    self.misses += 1  # if the flow is in the cache and has to be added again, then the impact is negative
                    self.miss_pi += 1  # if the flow is in the cache and has to be added again, then the impact is negative

        # if flows missed, divide the difference in time by misses
        if self.miss_pi != 0:
            self.avg_PI_IAT = (self.difference / self.miss_pi)

        # Set the first index in the state to Average PacketIn inter-arrival time
        self.state[0] = self.avg_PI_IAT

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)

        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        msg = ev.msg
        flow = str(msg.match)

        if flow in self.switches[msg.datapath.id]:
            del self.switches[msg.datapath.id][flow]

        self.fr_counter += 1  # increment by one every time a flow is removed

        self.total_dur += msg.duration_sec  # add the duration of the removed flow to the running total
        self.avg_fd = self.total_dur / self.fr_counter  # duration / flows

        # Set the second index in the state to avg_fd
        self.state[1] = self.avg_fd

    def dynamic_timeout(self):
        self.logger.info("Initiating RL process...")

        if self.total_pi != 0:
            self.hit = 1 - (self.misses / self.total_pi)  # % of flows for previously installed rules divided by the
            # total during polling period
        elif self.misses != 0:
            self.hit = 0
        else:
            self.hit = 1

        self.logger.info("Hit: %s", self.hit)

        # Collect switch data
        holder4 = [value for key, value in self.switch_data.items()]

        # Calculate use as median or set to 1 if data is missing
        self.use = np.median(holder4) if holder4 else np.nan
        # self.use = np.average(holder4) if holder4 else np.nan
        self.use = 1 if math.isnan(self.use) else self.use

        self.logger.info(self.use)

        # Update holders
        # Inverse of active flows
        self.holder1 += (1 - self.use) * 10
        # Inverse hit rate
        self.holder2 += (self.misses / self.total_pi) * 10 if self.total_pi != 0 else 0

        # Check if the episode is done
        done_bool = self.episode_step == 29

        # update running totals for episode step
        self.useSum += self.use
        self.hitSum += self.hit

        # calculate averages for state
        self.state[2] = (self.holder1 / (self.miniep + 1))
        self.state[4] = (self.holder2 / (self.miniep + 1))

        # average of totals
        self.avg_hit = (self.hitSum / (self.miniep + 1))
        self.avg_use = (self.useSum / (self.miniep + 1))

        self.logger.info("Average Hit: %s Average Use: %s", self.avg_hit, self.avg_use)
        self.logger.info("Packet In Messages: %s", self.total_pi)

        # reset values
        self.misses = 0
        self.total_pi = 0

        self.logger.info("Total Packet_In: %s", self.tp)

        if self.miniep >= 3:
            # Calculate reward
            # The minus sign (-) in the reward calculation ensures that the reward value is negative,
            # emphasizing that the aim is to minimize the deviation from the reference value. This approach
            # is particularly useful in reinforcement learning scenarios where the goal is to minimize errors or
            # deviations.
            reference_value = 1.7
            if self.holder1 != 0 and self.holder2 != 0:
                harmonic_mean = 2 / ((1 / (self.holder1 / 0.95)) + (1 / (self.holder2 / 0.95)))
                reward = -abs(reference_value - harmonic_mean)  # check math
            else:
                reward = 1


            # round values to nearest integer
            self.state = np.round(self.state, 1).astype(float)

            # if any values are larger than 10, set them equal to ten
            self.state = np.clip(self.state, None, 10)

            # set var equal timeout value's corresponding index so that the stored transition has proper information
            choice = self.action - 1

            if not evaluation_episode:
                self.model.train_on_transition(
                    torch.tensor(self.prev_state, dtype=torch.float32),
                    torch.tensor(choice, dtype=torch.long),
                    torch.tensor(self.state, dtype=torch.float32),
                    torch.tensor(reward, dtype=torch.float32),
                    torch.tensor(done_bool, dtype=torch.float32)
                )
            else:
                self.episode_reward += reward

            # set previous state equal to current state for replay value in next iteration
            self.prev_state = self.state

            new_action = (self.model.get_next_action(self.state, evaluation_episode)) + 1

            self.action = new_action

            self.logger.info("RL process completed...")

            self.switch_uses = []  # clear

"""
   def plot_results(self):
        n_results = len(self.evaluation_rewards)
        results_mean = [np.mean(self.evaluation_rewards[:i + 1]) for i in range(n_results)]
        results_std = [np.std(self.evaluation_rewards[:i + 1]) for i in range(n_results)]
        mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
        mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

        x_vals = list(range(len(results_mean)))
        x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]

        ax = plt.gca()
        ax.set_ylim([0, 200])
        ax.set_ylabel('Episode Score')
        ax.set_xlabel('Training Episode')
        ax.plot(x_vals, results_mean, label='Average Result', color='blue')
        ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
        ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
        ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
        plt.legend(loc='best')
        plt.show()

    def save_evaluation_reward(self, reward):
        with open('evaluation_rewards.txt', 'a') as file:
            file.write(f"{reward}\n")
        self.logger.info("Appended reward %s to evaluation_rewards.txt", reward)
"""
"""
TODO:
"""
