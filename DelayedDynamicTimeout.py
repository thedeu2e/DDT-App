# -----------------------------------------------------------
# (C) 2022 Nathan Harris, Jr., Greensboro, North Carolina
# Released under the MIT License (MIT)
# email ncharris1@aggies.ncat.edu
# -----------------------------------------------------------

# base classes/libraries
from ryu.app import simple_switch_13
import ryu.app.ofctl.api as ofctl_api
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet
from TD3 import TD3, utils

import numpy as np

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

STATE_DIM = 4  # 4-Dimensional State Space: [PPS, avg_fd, PIAT, action]
ACTION_DIM = 11  # 11-Dimensional Action Space: 0-10
MAX_ACTION = 10  # 10 is the choice with the highest value available to the agent
MAX_EPISODE_STEPS = 50000  # the maximum number of episodes used to train the model


class SimpleMonitor13(simple_switch_13.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        self.avg_fd = 0  # average flow duration from flow removed message | feature (state)
        self.curr_count = 0  # current number of flows in flow table from table stats reply
        self.fr_counter = 0  # running total of flows that have been removed from flow table from flow removed
        self.p_count = 0  # previous packet count from flow stats reply
        self.total_dur = 0  # running total of duration for flows removed from flow removed message
        self.hit = 0  # percentage of packets matched from table stats reply | outcome (reward)
        self.use = 0  # percentage of active flows from table stats reply | outcome (reward)
        self.PIAT = 0  # packet inter-arrival time from flow stats reply | feature (state)
        self.model = TD3.TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)  # TD3 initialization
        self.prev_state = np.array([None, None, None, None])  # placeholder for previous state
        self.state = np.array([None, None, None, None])  # placeholder for current state
        self.episode_step = 0  # episode counter initialization
        self.action = 10  # initial action | feature (state)
        self.replay_buffer = utils.ReplayBuffer(STATE_DIM, ACTION_DIM)  # Replay Buffer initialization

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        self.logger.info("starting flow monitoring thread")

        while True:
            if self.episode_step == 0:
                # Set initial state
                self.state = np.array([0, 0, 0, self.action], dtype=np.int8)
                self.prev_state = np.array([0, 0, 0, self.action])
            else:
                # Reset the state each time
                self.state = np.array([0, 0, 0, self.prev_state[3]], dtype=np.int8)

            for datapath in self.datapaths.values():
                self._request_stats(datapath)
                self.logger.info("requests sent")
                # self.send_barrier_request(datapath)
                # self.send_flow_mod(datapath)

            # self.send_barrier_request()

            self.logger.info("Current State%s", self.state)
            hub.sleep(self.action)

    def send_barrier_request(self, datapath):
        parser = datapath.ofproto_parser

        req = parser.OFPBarrierRequest(datapath)

        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def barrier_reply_handler(self, ev):
        self.logger.debug('OFPBarrierReply received')

    def add_flow(self, datapath, priority, match, actions, **kwargs):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # construct flow_mod message and send it.
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn,
                MAIN_DISPATCHER)  # Using 'MAIN_DISPATCHER' as the second argument means this function is called only after the negotiation completes
    def _packet_in_handler(self, ev):
        msg = ev.msg  # object that represents a packet_in data structure
        datapath = msg.datapath  # an object that represents a datapath (switch)
        ofproto = datapath.ofproto  # an object that represent the OpenFlow protocol that Ryu and the switch negotiated
        parser = datapath.ofproto_parser  # object that represents the OpenFlow protocol that Ryu & the switch negotiated

        # get Datapath ID to identify OpenFlow switches.
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # analyse the received packets using the packet library.
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        dst = eth_pkt.dst
        src = eth_pkt.src

        # get the received port number from packet_in message.
        in_port = msg.match['in_port']

        self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        # if the destination mac address is already learned,
        # decide which port to output the packet, otherwise FLOOD.
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        # construct action list.
        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time.
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions)

        # construct packet_out message and send it.
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  in_port=in_port, actions=actions,
                                  data=msg.data)
        datapath.send_msg(out)

    # Features request message
    # The controller sends a feature request to the switch upon session establishment.
    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        cookie = cookie_mask = 0
        msg1 = parser.OFPAggregateStatsRequest(datapath, 0,
                                               ofproto.OFPTT_ALL,
                                               ofproto.OFPP_ANY,
                                               ofproto.OFPG_ANY,
                                               cookie, cookie_mask,
                                               match)

        msg2 = parser.OFPTableStatsRequest(datapath, 0)

        # synchronize requests & replies so that thread waits for updates
        ofctl_api.send_msg(self, msg1, reply_cls=parser.OFPAggregateStatsReply, reply_multi=False)
        ofctl_api.send_msg(self, msg2, reply_cls=parser.OFPTableStatsReply, reply_multi=False)

        # Once all features are no longer set to None, fit our model on the sample
        self.dynamic_timeout()

    @set_ev_cls(ofp_event.EventOFPAggregateStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        results = ev.msg.body
        self.logger.info('%s', results)

        self.curr_count = results.flow_count  # flows in flow table for sample period

        if results.packet_count == 0:  # prevents zero division error
            PPS = results.packet_count / self.action  # packets per second = packets / duration | feature (state)
            self.PIAT = 0  # no packets arrived
        elif self.p_count == 0:  # if initial reply
            PPS = results.packet_count / self.action
            self.PIAT = self.action / results.packet_count  # packet inter-arrival time = duration / packets
        else:
            difference = abs(results.packet_count - self.p_count)  # calculate packet count
            PPS = difference / self.action
            if difference == 0:  # prevents zero division error
                self.PIAT = 0  # no packets arrived
            else:
                self.PIAT = self.action / difference

        # Set the first index in the state to PPS
        self.state[0] = PPS
        self.logger.info(self.state)
        # Set the third index in the state to PIAT
        self.state[2] = self.PIAT
        self.logger.info(self.state)

        self.p_count = results.packet_count  # hold value
        self.logger.info('PPS=%d', PPS)
        self.logger.info('PIAT=%f', self.PIAT)

    @set_ev_cls(ofp_event.EventOFPTableStatsReply, MAIN_DISPATCHER)
    def table_stats_reply_handler(self, ev):

        matched_sum = 0  # summation of packets matched in flow table(s)
        active_sum = 0  # summation of active flows in flow table(s)
        lookup_sum = 0  # summation of packets looked up in flow table(s)

        # stat collection
        for stat in ev.msg.body:
            active_sum += stat.active_count
            lookup_sum += stat.lookup_count
            matched_sum += stat.matched_count

        if lookup_sum == 0:  # prevents zero division error
            self.hit = 0
        else:
            self.hit = matched_sum / lookup_sum  # % of packets matched

        if self.curr_count == 0:  # prevents zero division error
            self.use = 0
        else:
            self.use = active_sum / self.curr_count  # % of flows actively receiving packets

        self.logger.info('TableStats: active flows=%d lookup=%d matched=%d', active_sum, lookup_sum, matched_sum)
        self.logger.info('Match Rate=%f Entry Use=%f', self.hit, self.use)

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        msg = ev.msg

        self.fr_counter = + 1  # increase by one every time a flow is removed

        self.total_dur += msg.duration_sec  # add the duration of the removed flow to the running total
        self.avg_fd = self.total_dur / self.fr_counter  # duration / flows

        # Set the second index in the state to avg_fd
        self.state[1] = self.avg_fd

    def send_flow_mod(self, datapath):
        ofp = datapath.ofproto
        ofp_parser = datapath.ofproto_parser

        idle_timeout = self.action
        match = ofp_parser.OFPMatch()
        actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
        inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                 actions)]
        req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                    idle_timeout,
                                    ofp.OFPP_ANY, ofp.OFPG_ANY,
                                    ofp.OFPFF_SEND_FLOW_REM,
                                    match, inst)
        datapath.send_msg(req)

        # self.barrier_reply_handler

    def dynamic_timeout(self):
        self.logger.info('here')

        # reward that agent receives for previous action places an emphasis on flows being active
        reward = (self.use + (self.hit * 0.5)) / 1.5

        done_bool = 1 if self.episode_step < MAX_EPISODE_STEPS else 0
        self.logger.info("here3")

        self.replay_buffer.add(self.prev_state, self.action, self.state, reward, done_bool)
        self.logger.info(self.replay_buffer)

        self.model.train(self.replay_buffer)

        self.logger.info("Previous State%s", self.prev_state)

        # increase episode counter
        self.episode_step += 1
        self.logger.info(self.episode_step)

        # Randomly select a new action
        new_action = np.argmax(self.model.select_action(self.state))
        if new_action != 0:
            self.action = new_action
            self.logger.info(self.action)

        # self.barrier_reply_handler
