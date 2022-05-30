# -----------------------------------------------------------
# (C) 2022 Nathan Harris, Jr., Greensboro, North Carolina
# Released under the MIT License (MIT)
# email ncharris1@aggies.ncat.edu
# -----------------------------------------------------------

# base classes/libraries
from operator import attrgetter

from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet
from ryu.ofproto.ofproto_v1_3 import OFPM_ALL
import TD3, utils
import random

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

STATE_DIM = 4
ACTION_DIM = 10
MAX_ACTION = 10
MAX_EPISODE_STEPS = 50000

class SimpleMonitor13(simple_switch_13.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        self.avgfd = 0
        self.curr_count = 0
        self.fr_counter = 0
        self.pcount = 0
        PPS = 0
        self.total_dur = 0
        self.hit = 0
        self.use = 0
        self.prev_pin = 0
        self.prev_dur = 0
        PIAT = 0
        self.model = TD3.TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)
        self.prev_state = [None, None, None, None]
        self.state = [None, None, None, None]
        self.episode_step = 0
        self.action = 5
        self.replay_buffer = utils.ReplayBuffer(STATE_DIM, ACTION_DIM)

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
            for dp in self.datapaths.values():
                self._request_stats(dp)
                # Collect the initial environment state in the prev_state variable
                # prev_state is a vector with 4 values
                if self.episode_step == 0:
                    self.prev_state = [None, 0, None, self.action]
                    # This while loop will keep running while one or more features are still set to none
                    while self.prev_state[0] is None or self.prev_state[2] is None:
                        continue
                # Once we've collected initial environment state, we collect the next state
                else:
                    # Reset the sample each time
                    self.state = [None, 0, None, self.prev_state[3]]
                    # This while loop will keep running while one or more features are still set to none
                    while self.state[0] is None or self.state[2] is None:
                        continue



                    # Once all features are no longer set to None, fit our model on the sample
                    reward = (self.use + (self.hit * 0.5)) / 1.5
                    done_bool = 1 if self.episode_step < MAX_EPISODE_STEPS else 0

                    self.replay_buffer.add(self.prev_state, self.action, self.state, reward, done_bool)

                    self.model.train(self.replay_buffer)

                    self.prev_state = self.state

                self.episode_step += 1

            # Randomly select a new action
            new_action = random.randint(0, 10)
            if new_action != 0:
                self.action = new_action


            # Submit request to change flow idle duration on switches
            for dp in self.datapaths.values():
                
                if self.action == 1:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 1
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 2:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 2
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 3:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 3
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 4:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 4
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 5:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 5
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 6:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 6
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 7:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 7
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 8:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 8
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 9:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 9
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)

                elif self.action == 10:
                    ofp = dp.ofproto
                    ofp_parser = dp.ofproto_parser

                    idle_timeout = 10
                    match = ofp_parser.OFPMatch()
                    actions = [ofp_parser.OFPActionOutput(ofp.OFPP_NORMAL, 0)]
                    inst = [ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                             actions)]
                    req = ofp_parser.OFPFlowMod(ofp.OFPFC_ADD,
                                                idle_timeout,
                                                ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                ofp.OFPFF_SEND_FLOW_REM,
                                                match, inst)
                    dp.send_msg(req)
                    
            hub.sleep(self.action)

            if self.episode_step >= MAX_EPISODE_STEPS:
                break

        self.model.save("model/TD3_trained")

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # construct flow_mod message and send it.
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER) #Using 'MAIN_DISPATCHER' as the second argument means this function is called only after the negotiation completes
    def _packet_in_handler(self, ev):
        msg = ev.msg #object that represents a packet_in data structure
        datapath = msg.datapath #an object that represents a datapath (switch)
        ofproto = datapath.ofproto #an object that represent the OpenFlow protocol that Ryu and the switch negotiated
        parser = datapath.ofproto_parser #an object that represent the OpenFlow protocol that Ryu and the switch negotiated
        
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
        
        
    #Features request message
    #The controller sends a feature request to the switch upon session establishment.

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        cookie = cookie_mask = 0

        req = parser.OFPAggregateStatsRequest(datapath, 0,
                                              ofproto.OFPTT_ALL,
                                              ofproto.OFPP_ANY,
                                              ofproto.OFPG_ANY,
                                              cookie, cookie_mask,
                                              match)

        datapath.send_msg(req)

        req = parser.OFPTableStatsRequest(datapath, 0)
        datapath.send_msg(req)

        req = parser.OFPMeterStatsRequest(datapath, 0, OFPM_ALL)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPAggregateStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        datapath = ev.msg.datapath
        results = ev.msg.body
        self.logger.info('%s', results)
        
        difference = 0
        self.curr_count = results.flow_count

        if self.pcount == 0:
            PPS = results.packet_count/10
        else:
            difference = abs(results.packet_count - self.pcount)
            PPS = difference/10

        # Set the first index in the sample to PPS
        self.state[0] = PPS

        self.pcount = results.packet_count
        self.logger.info(PPS)

    @set_ev_cls(ofp_event.EventOFPTableStatsReply, MAIN_DISPATCHER)
    def table_stats_reply_handler(self, ev):
        matched_sum = 0
        active_sum = 0
        lookup_sum = 0

        for stat in ev.msg.body:
            active_sum += stat.active_count
            lookup_sum += stat.lookup_count
            matched_sum += stat.matched_count

        self.hit = matched_sum/lookup_sum
        self.use = active_sum/self.curr_count
        self.logger.info('TableStats: active flows=%d lookup=%d matched=%d', active_sum, lookup_sum, matched_sum)
        self.logger.info('Match Rate=%f Entry Use=%f', self.hit, self.use)

    @set_ev_cls(ofp_event.EventOFPMeterStatsReply, MAIN_DISPATCHER)
    def meter_stats_reply_handler(self, ev):

        for stat in ev.msg.body:
            total_pin =+ stat.packet_in_count
            total_dur =+ stat.dur_sec + 1e9 * stat.duration_nsec

            if self.prev_pin == 0 & self.prev_dur == 0:
                PIAT = total_dur / total_pin
            elif abs(total_pin - self.prev_pin) != 0:
                diff_pin = abs(total_pin - self.prev_pin)
                diff_dur = abs(total_dur - self.prev_dur)
                PIAT = diff_dur / diff_pin

            self.prev_pin = total_pin
            self.prev_dur = total_dur
            self.logger.info(PIAT)

            self.state[2] = PIAT

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto

        self.fr_counter =+ 1

        if msg.reason == ofp.OFPRR_IDLE_TIMEOUT:
            reason = 'IDLE TIMEOUT'
        elif msg.reason == ofp.OFPRR_HARD_TIMEOUT:
            reason = 'HARD TIMEOUT'
        elif msg.reason == ofp.OFPRR_DELETE:
            reason = 'DELETE'
        elif msg.reason == ofp.OFPRR_GROUP_DELETE:
            reason = 'GROUP DELETE'
        else:
            reason = 'unknown'

        self.logger.debug('OFPFlowRemoved received: '
                      'cookie=%d priority=%d reason=%s table_id=%d '
                      'duration_sec=%d duration_nsec=%d '
                      'idle_timeout=%d hard_timeout=%d '
                      'packet_count=%d byte_count=%d match.fields=%s',
                      msg.cookie, msg.priority, reason, msg.table_id,
                      msg.duration_sec, msg.duration_nsec,
                      msg.idle_timeout, msg.hard_timeout,
                      msg.packet_count, msg.byte_count, msg.match)

        self.total_dur += msg.duration_sec
        self.avgfd = self.total_dur/self.fr_counter

        self.state[1] = self.avgfd
