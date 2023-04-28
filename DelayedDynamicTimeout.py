# -----------------------------------------------------------
# (C) 2023 Nathan Harris, Jr., Greensboro, North Carolina
# Released under the MIT License (MIT)
# email ncharris1@aggies.ncat.edu
# -----------------------------------------------------------

# base classes/libraries
from ryu.app import simple_switch_13
import ryu.app.ofctl.api as ofctl_api
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ether_types
from TD3 import TD3, utils

# required for Layer 4 matching
from ryu.lib.packet import in_proto
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import tcp
from ryu.lib.packet import udp

# required for misc. aspects of program
import numpy as np
import os
import random
from datetime import datetime

from cachetools import cached, TTLCache

# TD3 Model Parameters
STATE_DIM = 5  # 4-Dimensional State Space: [avg_PI_IAT, avg_fd, Inactive, action, misses]
ACTION_DIM = 10  # 10-Dimensional Action Space: 1-10
MAX_ACTION = 9  # 10 is the choice with the highest value available to the agent
MAX_EPISODES = 1000  # the maximum number of episodes used to train the model (300 second episodes * 1000 episdoes = 300,000 duration / 5 polling periods = 60,000 training steps)
MAX_EPISODE_STEPS = 14 # the maximum number of steps per episode (600 seconds/20 send polling intervals)

poll = 5 # polling incremnts in seconds


class SimpleMonitor13(simple_switch_13.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {} # dictionary of switches in network
        self.monitor_thread = hub.spawn(self._monitor)

        self.switches = {}  # a dictionary of switches within the network to keep track of key:flow rule entries, value:packet count pairs
        self.avg_PI_IAT = 0 # average packet in message inter-arrival time of misses | feature (state)
        self.avg_fd = 0  # average flow duration from flow removed message | feature (state)
        self.curr_count = 0  # current number of flows in flow table from table stats reply
        self.fr_counter = 0  # running total of flows that have been removed from flow table from flow removed
        self.total_dur = 0  # running total of duration for flows removed from flow removed message
        self.hit = 0  # percentage of packets matched from table stats reply | outcome (reward) | feature (state)
        self.use = 0  # percentage of active flows from table stats reply | outcome (reward) | feature (state)
        self.action = 10  # timeout value | feature (state)
        self.counter = 0 # total number of packet in request(s)
        self.cache = TTLCache(maxsize=1000, ttl=20) # cache where each item is accessbile for 10s
        self.misses = 0 # flow table misses
        self.difference = 0 # sum of packet in interarrival time diffrence
        self.total_pi = 0 # total count of packet_in messages
        self.hitSum = 0 # sum of hits
        self.useSum = 0 # sum of active
        self.avg_use = 0 # average use rate
        self.avg_hit = 0 # average hit rate
        self.pi_count = 0 # total packet in count for episode
        self.miss_pi = 0 # sum of missed packet in messages
        
        # RL Algorithm initialization specific 
        self.model = TD3.TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)  # TD3 initialization
        self.replay_buffer = utils.ReplayBuffer(STATE_DIM, ACTION_DIM)  # Replay Buffer initialization
        self.prev_state = np.array([None, None, None, None, None])  # placeholder for previous state
        self.state = np.array([None, None, None, None, None])  # placeholder for current state
        self.episode = 0 # episode counter intilization
        self.episode_step = 0  # episode step counter initialization
        self.miniep = 0  # miniepisodes
        self.decay_step = 0 # decay step
        
        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0 # exploration probability at start
        self.epsilon_min = 0.01 # minimum exploration probability
        self.epsilon_decay = 0.0005 # exponential decay rate for exploration prob

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
        

    def _monitor(self):
        self.logger.info("starting flow monitoring thread")

        while True:
            
            while self.episode < MAX_EPISODES:
                # reset episode variables
                self.episode_step = 0
                self.fr_counter = 0
                self.total_dur = 0
                self.difference = 0
                self.miss_pi = 0
                
                while self.episode_step < MAX_EPISODE_STEPS:
                    # reset episode step variables
                    self.miniep = 0
                    self.hitSum = 0
                    self.useSum = 0
                    
                    if self.episode_step == 0:
                        # initialize state and previous state array
                        self.state = np.array([0, 10, 0, self.action, 0], dtype=np.float)
                        self.prev_state = np.array([0, 0, 0, self.action, 0])
                    else:
                        # Reset the state each time
                        self.state = np.array([self.prev_state[0], self.prev_state[1], 0, self.action, self.prev_state[4]], dtype=np.float)

                    while self.miniep < 4:

                        # sends stats request to every switch
                        for datapath in self.datapaths.values():
                            self._request_stats(datapath)
                            self.send_barrier_request(datapath)
                        
                        # displays current episode, episode step, and ministep
                        self.logger.info("Episode: %s Step: %s Mini-Step: %s", self.episode, self.episode_step, self.miniep)

                        # displays current state of network
                        self.logger.info("Current State:%s ", self.state)
                        
                        # displays current timeout value
                        self.logger.info("timeout value: %s", self.action)
                        
                        # increment ministep
                        self.miniep += 1
                    
                        # thread sleeps for new duration selected by agent
                        hub.sleep(poll)
                    
                    # increment episode step
                    self.episode_step += 1
               
                # increment episode
                self.episode += 1

            # display saving message(s)
            self.logger.info("starting save")    
            self.model.save("DDTtrained")
            self.logger.info("finished save") 
            break
            # os._exit()

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
        self.total_pi += 1 # sum of packet_in messages during polling period
        
        msg = ev.msg  # object that represents a packet_in data structure
        datapath = msg.datapath  # an object that represents a datapath (switch)
        ofproto = datapath.ofproto  # an object that represent the OpenFlow protocol that Ryu and the switch negotiated
        parser = datapath.ofproto_parser  # object that represents the OpenFlow protocol that Ryu & the switch negotiated
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
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
                                            ip_proto=protocol,) #tcp_src=t.src_port, tcp_dst=t.dst_port, )

                #  If UDP Protocol
                elif protocol == in_proto.IPPROTO_UDP:
                    u = pkt.get_protocol(udp.udp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, eth_dst=dst, eth_src=src,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol, ) #udp_src=u.src_port, udp_dst=u.dst_port, )
                    

                # verify if we have a valid buffer_id, if yes avoid to send both
                # flow_mod & packet_out
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions)
                    
                flow = str(match) # create key
                
                if flow not in self.cache: # search for key in cache
                    self.cache[flow] = datetime.now() # if the flow isn't in the cache, add it with time it was added
                else:
                    now = datetime.now() # current time
                    self.difference += (now - self.cache[flow]).seconds # add difference of times to sum
                    self.cache[flow] = now # update flow's time
                    self.misses += 1 # if the flow is in the cahce and has to be added again, then the impact is negative
                    self.miss_pi += 1  # if the flow is in the cahce and has to be added again, then the impact is negative
        
        # if flows missed, divide the difference in time by misses
        if self.miss_pi != 0:
            self.avg_PI_IAT = (self.difference/self.miss_pi)
        
        # Set the first index in the state to Average PacketIn inter-arrival time
        self.state[0] = self.avg_PI_IAT
                    

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        
        datapath.send_msg(out)

    def send_barrier_request(self, datapath):
        ofp_parser = datapath.ofproto_parser

        req = ofp_parser.OFPBarrierRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def barrier_reply_handler(self, ev):
        self.logger.debug('OFPBarrierReply received')

    # Features request message
    # The controller sends a feature request to the switch upon session establishment.
    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        cookie = cookie_mask = 0

        flow = parser.OFPFlowStatsRequest(datapath, 0,
                                          ofproto.OFPTT_ALL,
                                          ofproto.OFPP_ANY, ofproto.OFPG_ANY,
                                          cookie, cookie_mask,
                                          match)

        # synchronize requests & replies so that thread waits for updates
        ofctl_api.send_msg(self, flow, reply_cls=parser.OFPFlowStatsReply, reply_multi=True)
        
        # Once all features are no longer set to None, fit our model on the sample
        self.dynamic_timeout()

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        self.curr_count = 0

        for stat in body:
            flow = str(stat.match) # create key

            # if flow not in dict, add flow and packet count and increment active flow count
            if flow not in self.switches[ev.msg.datapath.id]:
                self.switches[ev.msg.datapath.id][flow] = stat.packet_count
                self.curr_count += 1
            # if flow in dict, but packet count changed, update and increment active flow count
            elif self.switches[ev.msg.datapath.id][flow] != stat.packet_count:
                self.switches[ev.msg.datapath.id][flow] = stat.packet_count
                self.curr_count += 1
                
            # if list of flows not zero 
            if len(self.switches[ev.msg.datapath.id]) == 0: # prevents zero division error
                self.use = 0
            else:
                self.use = self.curr_count / len(self.switches[ev.msg.datapath.id]) # % of flows actively receiving packets

        self.logger.info("FC: %s", self.curr_count)
        self.logger.info("Total: %s", len(self.switches[ev.msg.datapath.id]))
        self.logger.info("Active: %s", self.use)

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
        
        if self.total_pi !=0:
            self.hit = 1-(self.misses / self.total_pi) # % of flows for previously installed rules divided by the total during polling period
        elif self.misses !=0:
            self.hit = 0
        else:
            self.hit = 1

        self.logger.info("Hit: %s", self.hit)
        
        # Inverse of active flows
        self.state[2] = (1 - (self.use)) * 10
        # Inverse of hit rate
        if self.total_pi != 0:
            self.state[4] = (self.misses / self.total_pi) * 10
        else:
            self.state[4] = 0
            
        # running totals for episode step
        self.useSum += self.use
        self.hitSum += self.hit
        
        # average of totals
        self.avg_hit = (self.hitSum / (self.miniep + 1))
        self.avg_use = (self.useSum / (self.miniep + 1))
        
        if self.avg_hit >= 0.80 and self.avg_use >= 0.80:
            reward = 9
        elif self.avg_hit <= 0.60 or self.avg_use <= 0.60:
            reward = -9
        else:
            reward = 0
            
        self.logger.info("Average Hit: %s Average Use: %s", self.avg_hit, self.avg_use)
        self.logger.info("Reward: %s", reward)
        
        # reset values
        self.misses = 0
        self.total_pi = 0

        done_bool = False if self.episode_step < MAX_EPISODE_STEPS and self.miniep <= 3 else True
        
        # set var equal timeout value's corresponding index so that the stored transition has proper information
        choice = self.action - 1
        
        # round values to nearest integer
        self.state = np.round(self.state)
        
        # if any values are larger than 10, set them equal to ten
        self.state = np.select([self.state >= 10], [10], self.state)
        
        # add transition to replay buffer
        self.replay_buffer.add(self.prev_state, choice, self.state, reward, done_bool)
        
        # if replay buffere has enough instances, train model
        if self.replay_buffer.size >= 256:
            self.model.train(self.replay_buffer)

        # set previous state equal to current state for replay value in next iteration
        self.prev_state = self.state

        if self.miniep >= 3 :

            # Randomly select a new action
            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
            
            self.logger.info("Probability: %s", explore_probability)

            if self.episode_step < 10 and explore_probability > np.random.rand():
                # increment decay step
                self.decay_step += 1
                # displays Q values for choices
                self.logger.info(self.model.select_action(self.state))
                # 
                choices = (np.flatnonzero(self.model.select_action(self.state) == np.max(self.model.select_action(self.state))))
                self.logger.info(choices)
                # Make a random action (exploration)
                # if hit rate is below 80%, a larger value will increase the hit rate
                if self.avg_hit < 0.80:
                    new_action = (random.randint(5, 9)+1)
                # if use rate is below 80%, a smaller value will increase the use rate
                elif self.avg_use < 0.80:
                    new_action = (random.randint(0, 4)+1)
                # model chooses highest Q values to maintain stability
                else:
                    new_action = (random.choice(choices)+1)  
            else:
                # Get action from Q-network (exploitation)
                # Estimate the Qs values state
                # Take the biggest Q value (= the best action)
                self.logger.info("not random")
                self.logger.info(self.model.select_action(self.state))
                choices = (np.flatnonzero(self.model.select_action(self.state) == np.max(self.model.select_action(self.state))))
                self.logger.info(choices)
                if len(choices) == 0:
                    new_action = 6
                else:
                    new_action =round(np.median(choices)+1)

            self.action = new_action

        self.barrier_reply_handler
   
