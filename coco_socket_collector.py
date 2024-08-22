#!/usr/bin/env python
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu import cfg

import csv
import os
import pickle
import socket
import socketserver
import time
import sys
import math
import socket
import logging
import json
import ast
from random import randint
import threading
from _thread import *

import numpy as np
import pandas as pd
from pandas.core.computation.check import NUMEXPR_INSTALLED
from scipy import interpolate
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from numpy.linalg import LinAlgError
from Cython.Compiler.Naming import self_cname

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import signal

host = '127.0.0.1'  # switch addr
port = 8888
collected_data = []
deviations_data = []
ts_fields_ = ['n_packets', 'n_bytes', 'duration']
ts_fields = ['time'] + ts_fields_ + ['value'] #y::value = fonction('n_packets', 'n_bytes', 'duration')

output_filename = "/home/shaw-cs/Documents/GitHub/DDT-App/Outputs/"
rep = output_filename
file_id = (time.strftime("%c")).replace(" ", "_")      # actual date and time
default_output_filename = rep + "COCO_CTRL_" + file_id +"_.csv"

# Monitoring metrics computation: network utilization
total_BW = 10000000  # 10 Mbps
def ts_function1(last_collected_n_bytes, actual_collected_n_bytes, talpha):
    #ts value y, monitoring time series stream depending on several flow stats
    # network utilization
    utilization = ((actual_collected_n_bytes - last_collected_n_bytes) * 8)/(talpha) # in %
    return utilization

def ts_function():
    utilization = 0
    return utilization
    

class COCOCollector(app_manager.RyuApp):
    def __init__(self, *args, **kwargs):
        super(COCOCollector, self).__init__(self, *args, **kwargs)
        
        CONF = cfg.CONF
        CONF.register_opts([
            cfg.FloatOpt('eta', default = 0, help = ('imprecision tolerance indicator')),
            cfg.FloatOpt('T0', default = 1, help = ('Time granularity - Initial period')),
            cfg.IntOpt('beta_init', default = 100, help = ('Initial beta')),
            cfg.IntOpt('beta', default = 10, help = ('beta')),
            cfg.IntOpt('W', default = 150, help = ('Sliding window size')),
            cfg.StrOpt('output_filename', default = default_output_filename, help = ('Simu Output filename'))])
        print("================= BEN COCO COLLECTOR MODULE =================")
        self.TIME_DEBUT = time.time()
        self.datapaths = {}
        
        self.ts_df = pd.DataFrame(columns=ts_fields) # Rolling time series data ==> Pandas DataFrame
        self.predicted_datapoints = []
        self.steps_ = 1  # number of future datapoints to forcast
        self.alpha = 0
        self.beta_init = CONF.beta_init
        self.beta = CONF.beta
        self.beta1 = self.beta_init
        self.gamma = 1
        self.deviation = 0
        self.num_deviations = 0
        self.ind = -1
        self.old_alpha = 0
        self.sample_id = 0
        self.t_indice = 0
                
        self.j = 0 #Number of samples collected with a specific alpha
        
        self.W = CONF.W # history window beta_init
        self.W_max = CONF.W #200
        self.eta = CONF.eta  # eta = to 0 fixed push, eta = 1 dafault mase threshold; < 1 more precise ; > 1 more flexible in term of accuracy
        self.MASE_th = self.eta  # default 1
        self.T0 = CONF.T0
        self.check_train = []
        self.last_observed = []
        self.last_forecasted = []
        self.last_observed_j = [] #debug
        self.last_forecasted_j = [] #debug
        self.H = []   # history with only collected datapoints and missing data are interpolated
        self.last_collected_n_bytes = 0
        self.model = None # ARIMA model; ExponentialSmoothing model ; ...
        self.arima_process_time = 0
        self.ARIMA_process_time_tab = []

        self.file_id = (time.strftime("%c")).replace(" ", "_")        
        #self.output_file_name = rep + "COCO_CTRL_" + self.file_id +"_.csv"
        self.output_file_name = CONF.output_filename
        
        self.coco_collector_socket = self.create_socket()
        
        print('************** eta={0}, T0={1}, beta0={2}, beta={3}, W={4}'.format(self.eta, self.T0, self.beta_init, self.beta, self.W_max))

        self.collection_initialization()
        self.coap_client_thread = hub.spawn(self.collect())

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                #self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                #self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    # This method should bind our socket to a host ip address and a port
    def create_socket(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((host, port))
        except socket.error:
            print('Failed to create socket')
            sys.exit()
        return s



    def collection_initialization(self):
        time_ = round((time.time() - self.TIME_DEBUT), 6)
        initialization_data = {"time": time_, "eta": self.eta, "t0": self.T0, "beta_init": self.beta_init, "beta": self.beta}
        msg = pickle.dumps(initialization_data)
        print("BEN =====================> CTRL expresses its interest ...")    
        
        # Print socket information
        local_addr = self.coco_collector_socket.getsockname()
        print(f"Server socket bound to: {local_addr}")

        self.coco_collector_socket.listen()
        # Wait for connections
        # Spawn a new thread each time a connection is received to handle the response and ongoing communication
        while True:
            c, addr = self.coco_collector_socket.accept()

            print ('Connected to :', addr[0], ':', addr[1])

            start_new_thread(self.connection_handler, (c, addr[0], msg))

    def connection_handler(self, connection, ipaddress, msg):
        print('Successful connection from ' + ipaddress)

        try:
            print("now")
            connection.sendall(msg)  # Tries to send the serialized data to a specified host and port
            print("now2")

            # Wait for confirmation
            confirmation = connection.recv(1024)

            if confirmation == "Initialization successful":
                print("Initialization confirmed.")


            else:
                print("Confirmation not received. Retrying...")

            self.fixed_collect(connection)

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

    def fixed_collect(self, connection):
        print("BEN==========>Collector fixed Collection")
        while True:
            data = connection.recv(1024)
            flow_table = pickle.loads(data)
            for flow_id, sample in flow_table.items():
                sample["receive_time"] = round((time.time() - self.TIME_DEBUT), 6)
                self.sample_id = self.sample_id + 1
                collected_data.append(sample)
                print('>>>>>> collected data == ', sample)
                print('++++ ctrl_sid = %d ctrl_alpha = %d \n' % (self.sample_id, self.alpha))
                value_ = sample['n_bytes']
                self.save_data_to_individual_COCO(self.sample_id, sample['n_packets'], sample['n_bytes'], sample['duration'], sample['idle_age'],
                                                  value_, sample['alpha'], sample['talpha'], self.arima_process_time, self.deviation, sample['absolute_time'])

    
    def adaptive_collect(self):
        print("BEN==========>Collector Adaptive Collection")
        while(1):
            d = self.coco_collector_socket.recvfrom(1024)
            data = d[0] #sample ==> dict
            sample = pickle.loads(data)
            sample["receive_time"] = round((time.time() - self.TIME_DEBUT), 6)
            self.sample_id = self.sample_id + 1
            self.t_indice = self.t_indice + 1
            self.j = self.j + 1 # new sample for actual alpha
            collected_data.append(sample)
            print('>>>>>> collected data == ', sample)
            print('++++ ctrl_sid = %d ctrl_alpha = %d \n' % (self.sample_id, self.alpha))  


                        
            # convert sample (info) to time series data 
            value_ = sample['n_bytes']
            self.last_collected_n_bytes = sample['n_bytes']
            new_pd = pd.DataFrame([[sample['time'], sample['n_packets'], sample['n_bytes'], sample['duration'], 
                                    value_]], columns=ts_fields)
            self.ts_df = pd.concat([self.ts_df, new_pd] , ignore_index=True, sort=False)
            self.H = self.H + self.interpolate(value_)  # slice to select element to avoid out of range for H = []
            print("++++++++++++++ check length H and ts_df", len(self.H)==len(self.ts_df),(len(self.H),len(self.ts_df)))
            self.last_observed.append(value_)
            self.last_observed_j.append(self.j) #debug
    
            self.old_alpha = self.alpha
            self.old_steps_ = self.steps_
    
            if (self.j == self.beta1): # last beta1 sample for alpha; beta1 = beta0 and beta
                if(self.beta1 == self.beta_init): # phase initiale
                    self.alpha = self.alpha+1
                    self.steps_ = self.alpha + 1
                    self.j=0
                    self.beta1 = self.beta
                    self.last_observed = []
                    self.old_alpha = 0 # cas particulier, gestion phase initiale
                    print("Change self.beta1 ==>", self.beta1)
                                            
                else:
                    #Apres la phase initiale : beta1 = beta
                    if(self.check_deviation()==0):
                        self.alpha = min(self.alpha + 1, 9) #9 alpha max
                    else:
                        print("OUPS ====> deviation Alarm RAISED")
                        self.deviation = 1
                        self.alpha = int(math.floor(self.alpha/(2*self.gamma)))
                        self.num_deviations = self.num_deviations + 1
                        
                        self.raise_deviation()
                        
                    self.steps_ = self.alpha + 1 #update
                    self.j=0

            self.save_data_to_individual_COCO(self.sample_id, sample['n_packets'], sample['n_bytes'], sample['duration'], sample['idle_age'],
                                              value_, sample['alpha'], sample['talpha'], self.arima_process_time, self.deviation, sample['absolute_time'])
            self.deviation = 0 #a revoir
                                   
            if(len(self.ts_df)>=(self.beta_init)):   # version d'atant +1   and (self.old_alpha != 0 )
                
                self.ARIMA_based_processing()
                
                #self.ExponentialSmoothing_based_processing("add")
                
                #self.ExponentialSmoothing_based_processing("mul")
            


            #self.old_alpha = self.alpha


    def interpolate(self, new_collected_value):
        last_value = self.H[len(self.H)-1:len(self.H)]
        print("+++ from interpolate (old_alpha,alpha)", (self.old_alpha, self.alpha))
        #if((self.old_alpha == 0) or (self.alpha == 0)):  # debug 12/11/2019
        if(self.alpha == 0):
            return [new_collected_value]
        else:
            last_value = last_value[0]
            alpha = self.alpha
            x = [0, alpha+1]
            y = [last_value, new_collected_value]
            f = interpolate.interp1d(x, y)
            xnew = range(alpha+2) # [0, 1, ..., alpha+1]
            ynew = f(xnew)   # use interpolation function returned by `interp1d`
            ynew = ynew.astype(int) 
            return ynew[1:len(ynew)].tolist()

    def interpolate1(self, last_value, new_collected_value, alpha):
        print("+++ from interpolate alpha =", alpha)
        if(alpha == 0):
            return [new_collected_value]
        else:
            last_value = last_value[0]
            x = [0, alpha+1]
            y = [last_value, new_collected_value]
            f = interpolate.interp1d(x, y)
            xnew = range(alpha+2) # [0, 1, ..., alpha+1]
            ynew = f(xnew)   # use interpolation function returned by `interp1d`
            ynew = ynew.astype(int) 
            return ynew[1:len(ynew)].tolist()


    # https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/TimeSeries/MASE.py
    def MASE(self, training_series, testing_series, prediction_series):
        """
        Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.

        See "Another look at measures of forecast accuracy", Rob J Hyndman

        parameters:
            training_series: the series used to train the model, 1d numpy array
            testing_series: the test series to predict, 1d numpy array or float
            prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
            absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.

        """
        #print ('Needs to be tested')
        n = training_series.shape[0]
        d = np.abs(  np.diff( training_series) ).sum()/(n-1)

        errors = np.abs(testing_series - prediction_series )
        return errors.mean()/d
  
    def check_deviation(self):
        #With MASE
        if(self.alpha == 0 ):  #self.old_alpha
            self.last_observed = []
            self.last_forecasted = []
            return (0)
        else:
            #print("MASE-check-dev>>>>>>>>>>>>>>> (last_observed, last_forecasted)", (self.last_observed, self.last_forecasted))
            #print("jjjjj >", (self.last_observed_j, self.last_forecasted_j))
            train = np.array(self.check_train)
            observed = np.array(self.last_observed)
            forecasted = np.array(self.last_forecasted)
            err_indicator = self.MASE(train, observed, forecasted)
            self.last_observed = []
            self.last_forecasted = []
            print("BEN check_deviation() =====> ERROR INDICATOR", err_indicator)
            if(err_indicator < self.MASE_th): # self.MASE_th = 1
                return (0)
            else:
                return (1)
            
            
    # check deviation with MAPE

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        #print("MAPE-func>>>>>>>>>>>>>>>>>>>>>>>>>>> (y_true, y_pred)", (y_true, y_pred))
        
        if(0 in y_true): # check if y_true contains zeros
            return 100  # un grand nombre pour dire qu'il y a un pb
        
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def check_deviation1(self):
        #with MAPE
        if(self.alpha == 0 ):  #self.old_alpha
            self.last_observed = []
            self.last_forecasted = []
            return (0)
        else:
            #print("MAPE-check-dev>>>>>>>>>>>>>>> (last_observed, last_forecasted)", (self.last_observed, self.last_forecasted))
            #print("jjjjj >", (self.last_observed_j, self.last_forecasted_j))
            err_indicator = self.mean_absolute_percentage_error(self.last_observed, self.last_forecasted) # = MAPE
            self.last_observed = []
            self.last_forecasted = []
            self.last_observed_j = []
            self.last_forecasted_j = []
            print("BEN check_deviation() =====> ERROR INDICATOR - MAPE", err_indicator)
            if(err_indicator < self.MASE_th): # self.MASE_th = 1
                return (0)
            else:
                return (1)
    
    
    # End check deviation with MASE
                                     
    def ARIMA_based_processing1(self):
        # Ze use here old_alpha instead of alpha because gamma msg sending if any, to avoid to much delay, we made checking send gamma or note before doing this processing 
        #that may consume a lot of time on the ctrl side when receiving sample fom the node
        print("BEN=======>  ARIMA Forcasting...")
        
        self.W = min(len(self.ts_df), self.W_max)
        
        arima_start_time = time.time()
        #history = self.ts_df.value.values
        #history = history[len(history)-self.W:len(history)]  # Effet window W HERE
        #history = self.ts_df.value.values[len(self.ts_df)-self.W:len(self.ts_df)]
        history = self.H[len(self.H)-self.W:len(self.H)]
        if((self.old_alpha != self.alpha) or (len(self.ts_df)==(self.beta_init+1))):
            print("BEN======> UPDATING ARIMA Model ==> NEW")
            self.model = auto_arima(history, start_p=1, start_q=0,
                             max_p=5, max_q=5, seasonal = False,
                             trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise
            self.check_train = history
        else:
            print("BEN======> Refreshing the ARIMA Model")
            self.model.update(history[int((len(history)-self.steps_)):len(history)]) # update with new last datapints
            self.check_train = history
                
        forecasts = self.model.predict(n_periods=int(self.steps_))  # uncomment this when we do not use nan handling algorithm
        #print("forecasts type ==", type(forecasts))
        
        #predictions give nans, try to handle with: https://github.com/tgsmith61591/pmdarima/issues/101
        
        # handle nan
        #history = history.astype('float64')
        #order = self.model.order
        #print('ARIMA based processing: >>>>> order = {0}'.format(order))
        #model = ARIMA(history, order=order)
        #model_fit = model.fit(disp=0)
        #forecasts = model_fit.forecast(steps = int(self.steps_))[0]
        # handle nan

        print("Forecasts ===>", forecasts)
        self.predicted_datapoints = []
        for i in range(int(self.steps_) - 1):
            self.predicted_datapoints.append(int(round(forecasts[i])))
            new_pd = pd.DataFrame([[self.ts_df.time[len(self.ts_df)-1]+i+1,int(round(forecasts[i]))]], columns=['time','value'])
            self.ts_df = pd.concat([self.ts_df, new_pd] , ignore_index=True, sort=False)
            self.save_predicted_data_to_individual_COCO(int(round(forecasts[i])))
        
        self.last_forecasted.append(int(round(forecasts[int(self.steps_) - 1]))) # append in last_forecasted only forecast that will be compared to bservations
        
        arima_end_time = time.time()
        self.arima_process_time = round((arima_end_time - arima_start_time), 2)  # en secondes
        self.ARIMA_process_time_tab.append(self.arima_process_time)


    def ARIMA_based_processing(self):
        
        #Consistent ARIMA, when nan occurred used simple interpolation
        
        # Ze use here old_alpha instead of alpha because gamma msg sending if any, to avoid to much delay, we made checking send gamma or note before doing this processing 
        #that may consume a lot of time on the ctrl side when receiving sample fom the node
        print("BEN=======>  ARIMA Forcasting...")
        
        self.W = min(len(self.ts_df), self.W_max)
        
        arima_start_time = time.time()

        history = self.H[len(self.H)-self.W:len(self.H)]
        if((self.old_alpha != self.alpha) or (len(self.ts_df)==(self.beta_init+1))):
            print("BEN======> UPDATING ARIMA Model ==> NEW")
            self.model = auto_arima(history, start_p=1, start_q=0,
                             max_p=5, max_q=5, seasonal = False,
                             trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise
            self.check_train = history
        else:
            print("BEN======> Refreshing the ARIMA Model")
            self.model.update(history[int((len(history)-self.steps_)):len(history)]) # update with new last datapints
            self.check_train = history
                
        forecasts = self.model.predict(n_periods=int(self.steps_))  # uncomment this when we do not use nan handling algorithm
        
        ## NaN ARIMA forecasts handling with interpolation from the 2 last datapoints
        
        if(math.isnan(forecasts[0])):
            print(">>>>>>> NAN forecasts handling")
            x = [0,1]
            y = history[len(history)-2:]
            f = interpolate.interp1d(x,y,fill_value="extrapolate")
            
            xnew = range(self.steps_+2)
            ynew = f(xnew)
            ynew = ynew.astype(int)
            
            forecasts = ynew[2:len(ynew)].tolist()
                    
        ## End NaN ARIMA forecasts handling with linear interpolation

        print("Forecasts ===>", forecasts)
        self.predicted_datapoints = []
        for i in range(int(self.steps_) - 1):
            self.predicted_datapoints.append(int(round(forecasts[i])))
            new_pd = pd.DataFrame([[self.ts_df.time[len(self.ts_df)-1]+i+1,int(round(forecasts[i]))]], columns=['time','value'])
            self.ts_df = pd.concat([self.ts_df, new_pd] , ignore_index=True, sort=False)
            self.save_predicted_data_to_individual_COCO(int(round(forecasts[i])))
        
        self.last_forecasted.append(int(round(forecasts[int(self.steps_) - 1]))) # append in last_forecasted only forecast that will be compared to observations
        
        arima_end_time = time.time()
        self.arima_process_time = round((arima_end_time - arima_start_time), 2)  # en secondes
        self.ARIMA_process_time_tab.append(self.arima_process_time)



    def ExponentialSmoothing_based_processing(self, trend_):
        #trend_ = 'add'; 'mul'
        
        # Ze use here old_alpha instead of alpha because gamma msg sending if any, to avoid to much delay, we made checking send gamma or note before doing this processing 
        #that may consume a lot of time on the ctrl side when receiving sample fom the node
        
        #tatsmodels.tsa.holtwinters.ExponentialSmoothing(endog, trend=None, damped=False, seasonal=None, seasonal_periods=None, dates=None, freq=None, missing='none')        
        #.fit(smoothing_level=None, smoothing_slope=None, smoothing_seasonal=None, damping_slope=None, optimized=True, use_boxcox=False, remove_bias=False, use_basinhopping=False, start_params=None, initial_level=None, initial_slope=None, use_brute=True)        
        #model_fit = model.fit(smoothing_level=0.9, smoothing_slope=0.6)
        #print((model_fit.params['smoothing_level'], model_fit.params['smoothing_slope']))
                
        print("BEN=======>  Exponential Smoothing forecasting Forcasting...")
        
        self.W = min(len(self.ts_df), self.W_max)
        
        arima_start_time = time.time()

        history = self.H[len(self.H)-self.W:len(self.H)]
        
        #phase de construction modele seulement a des debuts de chaque alpha; et des mises à jour à chaque nouveau datapoints enlevé. nouveau model a chaque prediction. aprioris temps d'execution tres faible pour Exponential smoothing
            
        self.check_train = history
        self.model = ExponentialSmoothing(history, trend=trend_)   # trend='mul'
                
        model_fit = self.model.fit()
                
        # make prediction
        forecasts = model_fit.predict(len(history), (len(history) + self.steps_ - 1))          

        print("Forecasts ===>", forecasts)
        self.predicted_datapoints = []
        for i in range(int(self.steps_) - 1):
            self.predicted_datapoints.append(int(round(forecasts[i])))
            new_pd = pd.DataFrame([[self.ts_df.time[len(self.ts_df)-1]+i+1,int(round(forecasts[i]))]], columns=['time','value'])
            self.ts_df = pd.concat([self.ts_df, new_pd] , ignore_index=True, sort=False)
            self.save_predicted_data_to_individual_COCO(int(round(forecasts[i])))
        
        self.last_forecasted.append(int(round(forecasts[int(self.steps_) - 1]))) # append in last_forecasted only forecast that will be compared to bservations
        self.last_forecasted_j.append(self.j)  #debug
        
        arima_end_time = time.time()
        self.arima_process_time = round((arima_end_time - arima_start_time), 2)  # en secondes
        self.ARIMA_process_time_tab.append(self.arima_process_time)

                

    def raise_deviation(self):
       
        print("BEN =====================> Deviation, gamma update ...")        
        time_ = round((time.time() - self.TIME_DEBUT), 6)
        deviation_data = {"time": time_, "gamma": 1}
        msg = pickle.dumps(deviation_data)        
        try :
            #Set the whole string
            self.coco_collector_socket.sendto(msg, (host, port))
        
        except (socket.error, msg):
            print('Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
            sys.exit()              

        
    # Enregistrement Output file
    #collect_: pour dire data collected and not forecated 
    def save_data_to_individual_COCO(self, sample_id_, n_packets_, n_bytes_, duration_, idle_age_, value_, alpha_, talpha_, ARIMA_time_, deviation_, absolute_time_):
        file_exists = os.path.isfile(self.output_file_name)
        line = []
        with open(self.output_file_name, 'a') as f:
            fieldnames = ['sample_id', 'n_packets', 'n_bytes', 'duration', 'idle_age', 'value', 'alpha', 'talpha', 'ARIMA_time', 'deviation', 'absolute_time', 'ctrl_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            # une boucle pr recuperer self.ts_df[self.ind]     
            line = {'sample_id' : sample_id_, 'n_packets' : n_packets_, 'n_bytes' : n_bytes_, 'duration' : duration_, 'idle_age' : idle_age_, 'value' : value_, 
                    'alpha' : alpha_, 'talpha' : talpha_, 'ARIMA_time' : self.arima_process_time, 'deviation' : deviation_, 'absolute_time' : absolute_time_, 'ctrl_time' : time.strftime("%c").split(" ")[3]}   
            writer.writerow(line)
            
    def save_predicted_data_to_individual_COCO(self,value_):
        file_exists = os.path.isfile(self.output_file_name)
        line = []
        with open(self.output_file_name, 'a') as f:
            fieldnames = ['sample_id', 'n_packets', 'n_bytes', 'duration', 'idle_age', 'value', 'alpha', 'talpha', 'ARIMA_time', 'deviation', 'absolute_time', 'ctrl_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # une boucle pr recuperer self.ts_df[self.ind]     
            line = {'value' : value_, 'ctrl_time' : time.strftime("%c").split(" ")[3]}   
            writer.writerow(line)

            
"""            
    def main():
        try:
            collector = COCOCollector()
            hub.joinall([collector.coap_client_thread])
        except Exception as e:
            print(f"Error: {e}")
        finally:
            collector.cleanup()

if __name__ == "__main__":
    main()
"""