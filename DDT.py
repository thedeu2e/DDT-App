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


# TD3 model paramters
STATE_DIM = 5  # 4-Dimensional State Space: [avg_ PI_IAT, avg_fd, PIAT, action, avg_PIAT]
ACTION_DIM = 10  # 10-Dimensional Action Space: 1-10
MAX_ACTION = 9  # 10 is the choice with the highest value available to the agent
MAX_EPISODES = 100  # the maximum number of episodes used to train the model
MAX_EPISODE_STEPS = 5000 # the maximum number of steps per episode

poll = 5 # polling incremnts in seconds


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
        self.avg_PI_IAT = 0 # average packet in message inter-arrival time of misses | feature (state)
        self.avg_PIAT = 0 # average packet inter-arrival time of flows that have timed out | feature (state)
        self.model = TD3.TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)  # TD3 initialization
        self.prev_state = np.array([None, None, None, None, None])  # placeholder for previous state
        self.state = np.array([None, None, None, None, None])  # placeholder for current state
        self.episode = 0 # episode counter intilization
        self.episode_step = 0  # episode step counter initialization
        self.action = 10  # timeout value | feature (state)
        self.counter = 0 # total number of packet in request(s)
        self.replay_buffer = utils.ReplayBuffer(STATE_DIM, ACTION_DIM)  # Replay Buffer initialization
        self.cache = TTLCache(maxsize=1000, ttl=20) # cache where each item is accessbile for 10s
        self.misses = 0 # table misses
        self.difference = 0 # sum of packet in interarrival time diffrence
        self.total_pi = 0 # total count of packet_in messages
        
        self.miniep = 0  # miniepisodes
        
        # Evaluation
        self.tp = 0 # total count of packet_in messages
        self.tpp = 0 # total count of polling periods
        self.tr = 0 # total sum of rewards
        self.ta = 0 # total sum of active rate percentage
        self.avg_reward = 0 # average reward = sum of rewards / total count of polling periods
        self.avg_active = 0 # average percentage of active flows = active rate / total count of polling periods

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
        
        self.model.load("DDTtrained")

        while True:
            #increment polling period
            self.tpp += 1
            
            # Reset the state each time
            self.state = np.array([self.prev_state[0], self.prev_state[1], None, self.action, self.prev_state[4]], dtype=np.float)
            
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
        self.tp += 1 # sum of packet_in messages
        
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
                    self.cache[flow] = time.time() # if the flow isn't in the cache, add it
                else:
                    now = time.time()
                    self.difference += now - self.cache[flow]
                    self.cache[flow] = now
                    self.misses += 1 # if the flow is in the cahce and has to be added again, then the impact is negative
        
        if self.misses != 0:
            self.avg_PI_IAT = (self.difference/self.misses)
        else:
            self.avg_PI_IAT = 0
        
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

        flows = parser.OFPAggregateStatsRequest(datapath, 0,
                                                ofproto.OFPTT_ALL,
                                                ofproto.OFPP_ANY,
                                                ofproto.OFPG_ANY,
                                                cookie, cookie_mask,
                                                match)

        # synchronize requests & replies so that thread waits for updates
        ofctl_api.send_msg(self, flow, reply_cls=parser.OFPFlowStatsReply, reply_multi=True)
        ofctl_api.send_msg(self, flows, reply_cls=parser.OFPAggregateStatsReply, reply_multi=True)

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
            self.hit = 1-(self.misses / self.total_pi) # % of flows for previously installed rules divided by the total during polling period
        elif self.misses !=0:
            self.hit = 0
        else:
            self.hit = 1

        self.logger.info("Hit: %s", self.hit)
        
        reward = ((self.use * 0.50) + (self.hit)) / 1.5
        
        self.tr += reward
        self.ta += self.use
        self.avg_reward = (self.tr/self.tpp)
        self.avg_active = (self.ta/self.tpp)
                
        self.misses = 0
        self.total_pi = 0

        self.logger.info("Average Active Rate: %s", self.avg_active)
        self.logger.info("Average Reward: %s", self.avg_reward)
        self.logger.info("Total Packet_In: %s", self.tp)

        # set previous state equal to current state for replay value in next iteration
        self.prev_state = self.state

        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        # Take the biggest Q value (= the best action)
        new_action = (np.argmax(self.model.select_action(self.state)) + 1)

        self.action = new_action
        
        self.logger.info("timeout value: %s", self.action)

        self.barrier_reply_handler
