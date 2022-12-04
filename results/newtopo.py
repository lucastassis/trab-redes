import atexit
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import info,setLogLevel
from mininet.link import TCLink
net = None

def createTopo():
    topo=Topo()
    #Create Nodes
    topo.addHost("h1", ip="10.0.0.10/24")
    topo.addHost("h2", ip="10.0.0.20/24")
    topo.addHost("h3", ip="10.0.0.30/24")
    topo.addSwitch('s1')
    topo.addHost('sta1', ip="10.0.0.1/24")
    topo.addHost('sta2', ip="10.0.0.2/24")
    topo.addHost('sta3', ip="10.0.0.3/24")
    #Create links
    topo.addLink('h1','s1')
    topo.addLink('h2','s1', loss=1)
    topo.addLink('h3','s1', loss=5)
    topo.addLink('sta1','s1')
    topo.addLink('sta2','s1')
    topo.addLink('sta3','s1')
    return topo

def startNetwork():
    topo = createTopo()
    global net
    net = Mininet(topo=topo, autoSetMacs=True, link=TCLink)
    net.start()
    CLI(net)

def stopNetwork():
    if net is not None:
        net.stop()

if __name__ == '__main__':
    atexit.register(stopNetwork)
    setLogLevel('info')
    startNetwork()
