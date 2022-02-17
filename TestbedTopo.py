# coding=UTF-8

from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch, OVSSwitch
from mininet.link import Link 
from mininet.cli import CLI
from mininet.log import setLogLevel, info

import time, os

Import numpy as np

class Testbed(object):
  
  def __init__(self):

    #Create Mininet instance
    self.net = Mininet()

    #Add the SDN controller to the network
    self.c0 = self.net.addController( name='Ryu', controller=RemoteController,  port=6633)
    
    #Add hosts(4) and switches(2) to the network
    self.h1 = self.net.addHost( name='host1', mac='00:00:00:00:00:01', ip='10.0.0.1' )
    self.h2 = self.net.addHost( name='host2', mac='00:00:00:00:00:02', ip='10.0.0.2' )
    self.h3 = self.net.addHost( name='host3', mac='00:00:00:00:00:03', ip='10.0.0.3' )
    self.h4 = self.net.addHost( name='host4', mac='00:00:00:00:00:04', ip='10.0.0.4' )
    self.s1 = self.net.addSwitch( name='switch1', cls=OVSKernelSwitch, protocols='OpenFlow13' )
    self.s2 = self.net.addSwitch( name='switch2', cls=OVSKernelSwitch, protocols='OpenFlow13' )

    #Create links between network nodes
    self.net.addLink(host1, switch1) 
    self.net.addLink(host3, switch1)
    self.net.addLink(host2, switch2) 
    self.net.addLink(host4, switch2)
    self.net.addLink(switch1, switch2)

    #Start execution.
    self.net.build()                           
    self.net.start()
    CLI(self.net)
    self.net.pingAll()
    
    #Generate network traffic
    self.h1.cmd("cd /usr/bin")
    self.h1.cmd("./ITGRecv")
    self.h2.cmd("cd /usr/bin")
    self.h2.cmd("./ITGRecv")
    self.h3.cmd("cd /usr/bin")
    self.h3.cmd("./ITGSend -a 10.0.0.2")
    self.h4.cmd("cd /usr/bin")
    self.h4.cmd("./ITGSend -a 10.0.0.1")
                               
  def cleanup(self):                                                            
    self.net.stop()
    
 if __name__ == '__main__':
  setLogLevel( 'info' )  # for CLI output
  execute()
