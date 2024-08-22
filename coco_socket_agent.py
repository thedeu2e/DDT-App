#!/usr/bin/env python3
import os
import pickle
import socket
import sys
import time
import math
#import pandas as pd
import threading
import hashlib

# Constants for server configuration
HOST = '127.0.0.1'    # Symbolic name meaning all available interfaces
PORT = 8888   # Arbitrary non-privileged port

# Constants related to time and threads
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_START_TIME = time.time()
threads = []

# Parameters for adaptive collection
beta_init = 50 #30
beta = 10
T=1

# FlowStats Data structure: time flowId flow_match cookie table actions n_packets n_bytes duration idle_age; alpha talpha

class COCOAgent():
    def __init__(self, switch, fixed_filename):
        self.eta = None
        self.coco_agent_socket = self.create_socket()
        self.ctrl_addr = ('127.0.0.1', 6633)    # will be get through first ctrl init data
        self.switch = switch
        
        self.alpha = 0
        self.talpha = T
        self.t0 = T
        self.alpha_max = 9
        self.beta_init = beta_init
        self.beta = beta
        self.gamma = 1
        self.deviation = 0
        self.num_deviations = 0
        self.ind = 0
        self.old_alpha = 0
        self.sample_id = 0
                
        self.j = 0 #Number of samples collected with a specific alpha 
        
        self.running = True
        #self.fixed_pd = pd.DataFrame(columns=['time', 'value'])
        self.fixed_filename = fixed_filename
        self.flow_data = {}  # Dictionary to store flow information
        self.switch_data = {} # Dictionary to store switch information

    def get_flow_stat(self, switch, flow_match):
        try:
            cmd_ = 'sudo ovs-ofctl dump-flows ' + switch + ' | grep cookie'
            result = os.popen(cmd_).readlines()
            
            # Update flow statistics in the switch_stats dictionary
            data = {
                'active': 0.0,
                'count': 0.0
                    }
            
            for line in result:
                # Define flow_stats and flow_stats_dict inside the loop
                flow_stats = line.split(", ")
                flow_stats_dict = dict(item.split("=") for item in flow_stats[:len(flow_stats)-1])
                
                if float(flow_stats_dict['idle_age'].split('s')[0]) < 5:
                    data['active'] += 1
                    
                data['count'] += 1                
            print("Final Data:", data)
                
            if data['count'] != 0.0:
                self.switch_data = (data['active']/data['count'])                
            else:
                self.switch_data = 0.0                

        except Exception as e:
            # Handle exceptions
            print(f"Error: {e}")
            
        print(self.switch_data)    
        return self.switch_data
    
    # Method to introduce sleep with support for periodic actions during the sleep
    def time_sleep(self, time_):
        try:
            if time_ == 0:
                return
            if time_ == self.t0:
                time.sleep(time_)
            else:
                ratio = int(time_ / self.t0) - 1
                z = 0
                while z < ratio:
                    time.sleep(self.t0)
                    self.get_flow_stat(self.switch, None)
                    z += 1

                time.sleep(self.t0)
        except Exception as e:
            print(f"Error in time_sleep: {e}")   
                
    # Method to create a socket for communication
    def create_socket(self):
        # Creation of  Datagram (udp) socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # print (s)
            print('Socket created')
        except (socket.error, msg):
            print('Failed to create socket. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
            sys.exit()
        
        return s
    
    # Cleanup method to close the socket
    def cleanup(self):
       # Print socket information before closing
        local_addr = self.coco_agent_socket.getsockname()
        print(f"Closing client socket bound to: {local_addr}")

        if hasattr(self, 'coco_agent_socket'):
            self.coco_agent_socket.close()
        print("Cleaning up and exiting gracefully...")
        sys.exit()
    
    # Method to send flow data as a sample
    def push_sample(self):
        try:
            sample = self.get_flow_stat(self.switch, None)
            msg = pickle.dumps(sample)
            self.coco_agent_socket.sendall(msg)           
        except Exception as e:
            print(f"Error in push_sample: {e}")

    # Method to stop the running loop
    def stop_running(self):
        try:
            self.running = False
        except KeyboardInterrupt:
            self.cleanup()
    # Method for fixed collection at a constant rate
    def fixed_collection(self):
        try:
            while self.running:
                #print("streaming ...  id= %d alpha= %d " % (self.sample_id + 1, self.alpha))
                self.push_sample()
                self.sample_id = self.sample_id + 1
                time.sleep(self.t0)
            self.cleanup()
        except KeyboardInterrupt:
            self.cleanup()
            
    # Method for adaptive collection with changing alpha values        
    def adaptive_collection(self):
        try:
            # Phase initiale beta0
            while(self.running and self.sample_id < self.beta_init):
                print("streaming ...  id= %d alpha= %d " % (self.sample_id+1, self.alpha))
                self.push_sample()
                self.sample_id = self.sample_id + 1      # sample number/id
                time.sleep(self.t0) #sample rate (period) => sampPeriod; here talpaha = 1 

            #transition
            self.beta1 = self.beta            
            self.alpha = self.alpha+1
            self.talpha = (1 + self.alpha)*self.t0
            self.j=0
            sample1 = self.get_flow_stat(self.switch, None)
            time.sleep(self.t0) #fait partie de la transition

            # After the initial beta phase   
            while self.running:
                print("streaming ...  id= %d alpha= %d " % (self.sample_id+1, self.alpha))

                self.push_sample()
                self.sample_id = self.sample_id + 1   
                self.j = self.j + 1

                if(self.j < self.beta):
                    self.time_sleep(self.talpha) #sample rate (period) => sampPeriod; here talpaha = 1                
                else:  # j == beta
                    if(self.alpha == 0):
                        self.alpha = min(self.alpha + 1, self.alpha_max) # incrementation
                        self.talpha = (1 + self.alpha)*self.t0
                        self.j=0
                        self.time_sleep(self.talpha)
                    else:  #alpha diff 0                    
                        start = time.time()
                        try:
                            #deviation alarm
                            self.coco_agent_socket.settimeout(1*self.t0) #2 par 1
                            d = self.coco_agent_socket.recvfrom(1024)
                            data = d[0]
                            deviation_data = pickle.loads(data)
                            print("++++++++++++++ Deviation Alarm message received, decreasing of alpha")
                            self.alpha = math.floor(self.alpha/(2*deviation_data["gamma"])) # decreasing
                            self.talpha = (1 + self.alpha)*self.t0
                            self.j=0
                            end = time.time()

                            time.sleep(max(0,(self.t0 - (end-start)))) # suppose que end - start < t0

                            if(self.alpha > 0): # si 0, on sort et on fait le push directement
                                sample1 = self.get_flow_stat(switch, flow_match)

                                self.time_sleep(max(0,(self.talpha - self.t0)))                     
                            #self.time_sleep(self.talpha - (end-start))                     
                        # except TimeoutException:
                        except socket.timeout:
                            print("++++++++++++++ Timeout!!! No deviation alarm Incrementation of alpha")
                            self.alpha = min(self.alpha + 1, self.alpha_max) # incrementation
                            self.talpha = (1 + self.alpha)*self.t0
                            self.j=0
                            ##
                            sample1 = self.get_flow_stat(self.switch, None)
                            ##
                            self.time_sleep(max(0,(self.talpha - 1*self.t0)))  # minus the same as settimeout
                            continue
            self.cleanup()
        except KeyboardInterrupt:
            self.cleanup()

    # Main method to run the COCOAgent
    def run(self):
        # Print socket information
        self.coco_agent_socket.connect((HOST, PORT))
        local_addr = self.coco_agent_socket.getsockname()
        print(f"Client socket bound to: {local_addr}")
        
        try:
            # Collection initialization from the controller
            data = self.coco_agent_socket.recv(1024)
            init_data = pickle.loads(data)
            self.ctrl_addr = HOST
            print("Collection initialized : ", init_data)

            confirmation_message = "Initialization successful"
            self.coco_agent_socket.sendall(self.switch.encode('utf-8'))

            self.eta = init_data["eta"]
            self.t0 = init_data["t0"]
            self.beta_init = init_data["beta_init"]
            self.beta = init_data["beta"]

            if(self.eta == 0):
                self.fixed_collection()
            else:
                self.adaptive_collection() 
        except KeyboardInterrupt:
            print("KeyboardInterrupt at running. Stopping gracefully...")
            self.cleanup()


if __name__ == '__main__':
    print("+++++++++ BEN DEBUT +++++++++")
    switch = sys.argv[2]  # Assuming the switch name is the third argument
    print(switch + " Socket Server .....")
    fixed_filename = sys.argv[4]  # Assuming the fixed filename is the fifth argument
    flow_match = None
    TIME_DEBUT = time.time()
    coco_agent = COCOAgent(switch,fixed_filename)
        
    try:
        coco_agent.run()
        print(switch + " Agent Initiated .....")
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Stopping gracefully...")
        self.stop_running()