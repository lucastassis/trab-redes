import atexit
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import info,setLogLevel
from mininet.link import TCLink
from mininet.nodelib import NAT
net = None

def createTopo():
	topo=Topo()
	gwLocalIP = "10.0.0.1"
	#Create Nodes
	topo.addHost("serv", cls=NAT, inetIntf="enp0s3", inNamespace=False)
	#inetIntf eh o nome da interface de rede NA VM (fora do Mininet) que tem acesso 
	#ao seu notebook e a Internet
	topo.addHost("h1", defaultRoute='via %s' % gwLocalIP) #IP do h1
	topo.addHost("h2", defaultRoute='via %s' % gwLocalIP) #IP do h1
	topo.addHost("h3", defaultRoute='via %s' % gwLocalIP) #IP do h1
	#Create Switches
	topo.addSwitch('sw0')
	topo.addSwitch('sw1')
	topo.addSwitch('sw2')
	topo.addSwitch('sw3')
	#Create links
	topo.addLink('h1','sw1', bw=10, loss=1)
	topo.addLink('h2','sw2', bw=50, loss=0)
	topo.addLink('h3','sw3', bw=100, loss=0)
	topo.addLink('serv','sw0')
	topo.addLink('sw0','sw1')
	topo.addLink('sw0','sw2')
	topo.addLink('sw0','sw3')
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

