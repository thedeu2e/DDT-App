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
MAX_EPISODE_STEPS = 50000  # the maximum number of episodes used to train the model

poll = 3


class SimpleMonitor13(simple_switch_13.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
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
        self.episode_step = 0  # episode counter initialization
        self.action = 10  # initial action | feature (state)
        self.holder = 0 # holds value
        self.t1 = 0 # holds time value 
        self.t2 = 0 # holds time value
        self.counter = 0 # total number of packet in request(s)
        self.difference = 0
        self.sum = 0 #sum of time differnce
        self.replay_buffer = utils.ReplayBuffer(STATE_DIM, ACTION_DIM)  # Replay Buffer initialization

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
        
        counter = 0

        while True:
            if self.episode_step == 0:
                # initialize state and previous state array
                self.state = np.array([None, 10, None, self.action, None], dtype=np.float)
                self.prev_state = np.array([0, 0, 0, self.action, 0])
            else:
                # Reset the state each time
                self.state = np.array([self.prev_state[0], self.prev_state[1], None, self.action, self.prev_state[4]], dtype=np.float)

            # sends stats request to every switch
            for datapath in self.datapaths.values():
                self._request_stats(datapath)
                self.send_barrier_request(datapath)

            # displays current state of network
            self.logger.info("Current State:%s ", self.state)
            counter += 1
            
            if counter == 50000:
                self.model.save("DDTtrained")
                os.exit

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
        self.t1 = time.perf_counter()
        self.counter += 1
        
        if self.counter > 1:
            self.sum += (self.t1 - self.t2)
            self.avg_PI_IAT = (self.sum / self.counter)
            self.t2 = self.t1
            
            # Set the first index in the state to Average PI IAT
            self.state[0] = self.avg_PI_IAT
        else:
            self.t2 = self.t1
        
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
        self.logger.info("Out: %s", self.fr_counter)

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
        self.avg_PIAT = self.total_frpackets /self.avg_fd #average PIAT of flows that have timed out

        # Set the second index in the state to avg_fd
        self.state[1] = self.avg_fd
        # Set the third index in the state to PIAT
        self.state[4] = self.avg_PIAT

    def dynamic_timeout(self):
        
        # reward that agent receives for previous action places an emphasis on flows being active
        if self.curr_count == self.holder:
            reward = (self.use + (self.hit * 0.5)) / 1.5
        elif self.curr_count > self.holder:
             reward = ((self.use + (self.hit * 0.5)) / 1.5) + 1
        else:
             reward = ((self.use + (self.hit * 0.5)) / 1.5) - 1
                
        self.holder = self.curr_count

        self.logger.info("Reward: %s", reward)

        done_bool = 1 if self.episode_step < MAX_EPISODE_STEPS else 0

        self.replay_buffer.add(self.prev_state, self.action, self.state, reward, done_bool)

        self.model.train(self.replay_buffer)

        # self.logger.info("Previous State%s", self.prev_state)

        # set previous state equal to current state for replay value in next iteration
        self.prev_state = self.state

        # increase episode counter
        self.episode_step += 1
        self.logger.info("Step: %s", self.episode_step)

        # Randomly select a new action
        new_action = (np.argmax(self.model.select_action(self.state)) + 1)

        self.action = new_action

        self.logger.info("time: %s", self.action)

        self.barrier_reply_handler
        
