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

import numpy as np
import os
import time
import random

from cachetools import cached, TTLCache
from datetime import datetime

"""""
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=50000, type=int) # replay buffer size
parser.add_argument('--num_iteration', default=100000, type=int) #  num of  games
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)
# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()
"""

STATE_DIM = 5  # 4-Dimensional State Space: [avg_ PI_IAT, avg_fd, PIAT, action, avg_PIAT]
ACTION_DIM = 10  # 10-Dimensional Action Space: 1-10
MAX_ACTION = 9  # 10 is the choice with the highest value available to the agent
MAX_EPISODES = 100  # the maximum number of episodes used to train the model
MAX_EPISODE_STEPS = 5000 # the maximum number of steps per episode

poll = 10 # polling incremnts in seconds


class SimpleMonitor13(simple_switch_13.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {} # list of switches in network
        self.monitor_thread = hub.spawn(self._monitor)

        self.switches = {}  # a list of switches within the network to keep track of key:flow rule entries, value:packet count pairs
        self.avg_fd = 0  # average flow duration from flow removed message | feature (state)
        self.curr_count = 0  # current number of flows in flow table from table stats reply
        self.fr_counter = 0  # running total of flows that have been removed from flow table from flow removed
        self.p_count = 0  # previous packet count from flow stats reply
        self.total_packets = 0 # holds total packets in switch at time of polling
        self.total_frpackets = 0 # holds total of packets that has been removed
        self.total_dur = 0  # running total of duration for flows removed from flow removed message
        self.hit = 0  # percentage of packets matched from table stats reply | outcome (reward)
        self.use = 0  # percentage of active flows from table stats reply | outcome (reward)
        self.PIAT = 0  # packet inter-arrival time from flow stats reply | feature (state)
        self.avg_PI_IAT = 0 # average packet in message inter-arrival time | feature (state)
        self.avg_PIAT = 0 # average packet inter-arrival time of flows that have timed out | feature (state)
        self.model = TD3.TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)  # TD3 initialization
        self.prev_state = np.array([None, None, None, None, None])  # placeholder for previous state
        self.state = np.array([None, None, None, None, None])  # placeholder for current state
        self.episode = 0 # episode counter intilization
        self.episode_step = 0  # episode step counter initialization
        self.action = 10  # initial action | feature (state)
        self.holder = 0 # holds value
        self.t1 = 0 # holds time value 
        self.t2 = 0 # holds time value
        self.counter = 0 # total number of packet in request(s)
        self.c2 = 0 # temperarily holds value of packet_in messages per polling period.
        self.messages = 0 # hold c2 value
        self.sum = 0 #sum of time differnce
        self.replay_buffer = utils.ReplayBuffer(STATE_DIM, ACTION_DIM)  # Replay Buffer initialization
        self.cache = TTLCache(maxsize=100, ttl=20) # cache where each item is accessbile for 10s
        self.r = 0 # value for reward function
        self.total_pi = 0 # total count of packet_in messages
        
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
        
        self.trainedModel = self.model.load("DDTtrained")

        while True:           
             # sends stats request to every switch
             for datapath in self.datapaths.values():
             self._request_stats(datapath)
             self.send_barrier_request(datapath)

             # displays current state of network
             self.logger.info("Current State:%s ", self.state)
                    
             # thread sleeps for new duration selected by agent
             hub.sleep(poll)
           

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
        self.t1 = time.perf_counter() # record time function is called
        self.counter += 1 # increase packet in message counter
        self.c2 += 1
        
        
        if self.counter > 1: # if the counter has been called more than once
            self.sum += (self.t1 - self.t2) # subtract the current time from the previous time and add difference to the sum of all differences between function calls
            self.avg_PI_IAT = (self.sum / self.counter) # divide sum of all differences by total number of calls for average
            self.t2 = self.t1 # set t2 equal to t1 for next function call
            
            # Set the first index in the state to Average PacketIn inter-arrival time
            self.state[0] = self.avg_PI_IAT
        else:
            self.t2 = self.t1 # set t2 equal to t1 for next function call
        
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
                    self.cache[flow] = flow # if the flow isn't in the cache, add it
                    self.r += 1
                #else:
                    #self.r -= 1 # if the flow is in the cahce and has to be added again, then the impact is negative
                    

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

        flows = parser.OFPAggregateStatsRequest(datapath, 0,
                                                ofproto.OFPTT_ALL,
                                                ofproto.OFPP_ANY,
                                                ofproto.OFPG_ANY,
                                                cookie, cookie_mask,
                                                match)

        table = parser.OFPTableStatsRequest(datapath, 0)

        # synchronize requests & replies so that thread waits for updates
        ofctl_api.send_msg(self, flow, reply_cls=parser.OFPFlowStatsReply, reply_multi=True)
        ofctl_api.send_msg(self, flows, reply_cls=parser.OFPAggregateStatsReply, reply_multi=True)
        ofctl_api.send_msg(self, table, reply_cls=parser.OFPTableStatsReply, reply_multi=True)

        # Once all features are no longer set to None, fit our model on the sample
        self.dynamic_timeout()

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        self.curr_count = 0

        for stat in body:
            flow = str(stat.match)

            if flow not in self.switches[ev.msg.datapath.id]:
                self.switches[ev.msg.datapath.id][flow] = stat.packet_count
                self.curr_count += 1
            elif self.switches[ev.msg.datapath.id][flow] != stat.packet_count:
                self.switches[ev.msg.datapath.id][flow] = stat.packet_count
                self.curr_count += 1

        self.logger.info("FC: %s", self.curr_count)
        self.messages = self.c2
        self.c2 = 0
        #  self.logger.info(self.switches)

    @set_ev_cls(ofp_event.EventOFPAggregateStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        results = ev.msg.body

        self.total_packets = results.packet_count
        self.logger.info('%s', results)

        if results.packet_count == 0:  # prevents zero division error
            self.PIAT = 0  # no packets arrived
        elif self.p_count == 0:  # if initial reply
            self.PIAT = (results.flow_count * 0.002) / results.packet_count  # packet inter-arrival time = duration / packets
        else:
            difference = abs(results.packet_count - self.p_count)  # calculate packet count
            if difference == 0:  # prevents zero division error
                self.PIAT = 0  # no packets arrived
            else:
                self.PIAT = (results.flow_count * 0.002) / difference

        if results.flow_count == 0:  # prevents zero division error
            self.use = 0
        elif self.curr_count > results.flow_count: # if flows timeout between the stats reply
            self.use = results.flow_count / self.curr_count
        else:
            self.use = self.curr_count / results.flow_count  # % of flows actively receiving packets

        self.logger.info("Active: %s", self.use)
        # Set the third index in the state to PIAT
        self.state[2] = self.PIAT

        self.p_count = results.packet_count  # hold value
        self.logger.info("Total: %s", results.flow_count)

    @set_ev_cls(ofp_event.EventOFPTableStatsReply, MAIN_DISPATCHER)
    def table_stats_reply_handler(self, ev):

        lookup_sum = 0  # summation of packets looked up in flow table(s)
        matched_sum = 0 # summation of packets matched to existing flow rule entries in flow table(s)

        # stat collection
        for stat in ev.msg.body:
            lookup_sum += stat.lookup_count
            matched_sum += stat.matched_count

        if lookup_sum == 0:  # prevents zero division error
            self.hit = 0
        else:
            self.hit = matched_sum / lookup_sum  # % of packets matched

        self.logger.info("Hit: %s", self.hit)

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        msg = ev.msg
        flow = str(msg.match)
        
        if flow in self.switches[msg.datapath.id]:
            del self.switches[msg.datapath.id][flow]

        self.fr_counter += 1  # increment by one every time a flow is removed
        self.total_frpackets += msg.packet_count # sum of packets transmitted by flows that timeout

        self.total_dur += msg.duration_sec  # add the duration of the removed flow to the running total
        self.avg_fd = self.total_dur / self.fr_counter  # duration / flows
        if self.total_frpackets == 0:
            self.avg_PIAT = None
        else:
            self.avg_PIAT = self.avg_fd / self.total_frpackets #average PIAT of flows that have timed out

        # Set the second index in the state to avg_fd
        self.state[1] = self.avg_fd
        # Set the third index in the state to PIAT
        self.state[4] = self.avg_PIAT
        
        #self.logger.info("Out: %s", self.fr_counter)

    def dynamic_timeout(self):
        
        if self.total_pi !=0:
            pin = self.r / self.total_pi # packet_in messages for new  flows divided by the total
        else:
            pin = 0
        
        reward = ((self.use * 0.50) + (self.hit * 0.50) + pin ) / 2.0
                
        self.r = 0
        self.total_pi = 0

        self.logger.info("Reward: %s", reward)

        # set previous state equal to current state for replay value in next iteration
        #self.prev_state = self.state

        # increase episode counter
      
        #self.logger.info("Episode: %s Step: %s", self.episode, self.episode_step)

    
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        # Take the biggest Q value (= the best action)
        new_action = (np.argmax(self.trainedModel.select_action(self.state)) + 1)

        self.action = new_action

        self.logger.info("timeout value: %s", self.action)

        self.barrier_reply_handler
