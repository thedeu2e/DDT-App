from mininet.topo import Topo


class LinearTopo(Topo):

    def __init__(self, n=2):
        # Initialize topology
        Topo.__init__(self)
        h = []
        s = ['s1']
        self.addSwitch('s1')

        for i in range(1, n+1):
            str_h = 'h' + str(i)
            h.append(self.addHost(str_h))

            self.addLink(h[-1], s[-1])


topos = {'linear': (lambda: LinearTopo())}
