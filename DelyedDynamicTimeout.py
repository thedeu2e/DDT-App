from operator import attrgetter
from datetime import datetime
from ryu.app import simple_switch
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

sys.path.insert(0, './TD3')
sys.path.insert(0, './utils')

from TD3 import Actor as Actor 
from TD3 import Critic as Critic 
from utils import ReplayBuffer as Memory

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

class SimpleMonitor(simple_switch.SimpleSwitch):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.monitor_thread = hub.spawn(self._monitor)
            
        self.datapaths = {}
        self.state = {}
        self.unrolled_state = []
        self.input_state = []
        
        # values for packet calculation
        self.flow_packet_count = {}
        
        # to calculate deltas for bandwith usage calculation
        self.flow_byte_counts = {}
        
        # to calculate deltas for bandwith usage calculation
        self.port_byte_counts = {}
        
        self.reward = 0.0
       
        self.fields = {'time':'','datapath':'','in-port':'','eth_src':'','eth_dst':'','out-port':'','total_packets':0,'total_bytes':0}

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.state[datapath.id] = []
                self.datapaths[datapath.id] = datapath
                
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        self.logger.info('time\tdatapath\tin-port\teth-src\teth-dst\tout-port\ttotal_packets\ttotal_bytes')
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5)
            
    def get_state(self):
        for dp in self.datapaths.values():
            self.send_flow_stats_request(dp)
        hub.sleep(2 ) #TODO sleep
        self.format_state()  # TODO
        self.calculate_reward()
        
    # Convert from data to bitrate(Kbps)
    @staticmethod
    def bitrate(self,data):
        return round(float(data * 8.0 / (self.interval*1000)),2) 

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        
        flow_count = 0

        for stat in sorted([flow for flow in body if flow.priority == 1],
                           key=lambda flow: (flow.match['in_port'],
                                             flow.match['eth_dst'])):
            flow_count += 1
            
            #print details of flows
            self.fields['time'] = datetime.utcnow().strftime('%s')
            self.fields['datapath'] = ev.msg.datapath.id
            self.fields['in_port'] = stat.match['in_port']
            self.fields['eth_src'] = stat.match['eth_src']
            self.fields['eth_dst'] = stat.match['eth_dst']
            self.fields['out_port'] = stat.instructions[0].actions[0].port
            
            self.logger.info('data\t%s\t%x\t%x\t%s\t%s\t%x\t%d\t%d',self.fields['time'],self.fields['datapath'],self.fields['duration_sec'],self.fields['idle_timeout'],self.fields['in-port'],self.fields['eth_src'],self.fields['eth_dst'],self.fields['out-port'],self.fields['total_packets'],self.fields['total_bytes'])
            
            # Check if we have a previous reading for this flow
            # Calculate packet increase over the last polling interval
            difference = 0
            key = (self.fields['datapath'], self.fields['in_port'], self.fields['eth_src'], self.fields['eth_dst'], self.fields['out_port'])
            if key in self.flow_packet_count:
                pcount = self.flow_packet_count[key]
                difference = (stat.packet_count - pcount)
            self.flow_packet_count[key] = stat.packet_count
            
            #Calculate bandwith usage over the last polling interval
            rate = 0
            if key in self.flow_byte_counts:
                bcount = self.flow_byte_counts[key]
                rate = self.bitrate(self,stat.byte_count - bcount)
            self.flow_byte_counts[key] = stat.byte_count
           
        if len(self.state[datapath.id]) == 0:
            self.state[datapath.id].append({})
            self.state[datapath.id].append(difference)
            self.state[datapath.id].append(rate)
            self.state[datapath.id].append(flow_count)
        else:
            self.state[datapath.id][1] = difference
            self.state[datapath.id][2] = rate
            self.state[datapath.id][3] = flow_count
            
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        temp = []
        
        for stat in sorted(body, key=attrgetter('port_no')):
             if stat.port_no != ofproto_v1_3.OFPP_LOCAL:
                key = (ev.msg.datapath.id, stat.port_no)
                rx_bitrate, tx_bitrate = 0, 0
                total_Kbps=0
                if key in self.port_byte_counts:
                    cnt1, cnt2 = self.port_byte_counts[key]
                    rx_bitrate = self.bitrate(self,stat.rx_bytes - cnt1)
                    tx_bitrate = self.bitrate(self,stat.tx_bytes - cnt2)
                    total_Kbps= rx_bitrate + tx_bitrate
                self.port_byte_counts[key] = (stat.rx_bytes, stat.tx_bytes)
                
             temp.append(str(stat.rx_packets))
             temp.append(str(stat.rx_bytes))
             temp.append(str(stat.rx_bitrate))
             temp.append(str(stat.tx_bitrate))
             temp.append(str(stat.tx_packets))
             temp.append(str(stat.tx_bytes))
             temp.append(str(total_Kbps))
             self.state[datapath.id][0][stat.port_no] = temp
    
    def format_state(self):
    def get_reward(self):
    def reset(self):            
    def step(self,action):
          return next_state,reward,done
        
    def main(self):
