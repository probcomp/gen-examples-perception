import rpyc
import bpy

class BlockingServer(rpyc.utils.server.Server):

    def _accept_method(self, sock):
        self._authenticate_and_serve_client(sock)

def addNode(tree, typ):
    node = tree.nodes.new(typ)
    return node

def linkNodes(tree, node1, node2, out=0, inp=0):
    tree = bpy.context.scene.node_tree
    tree.links.new(node1.outputs[out], node2.inputs[inp])


def setup():
    bpy.data.scenes['Scene'].use_nodes = True
    bpy.data.scenes['Scene'].render.layers['RenderLayer'].use_pass_normal = True # ?
    tree = bpy.context.scene.node_tree
    addNode(tree, 'CompositorNodeNormalize')
    render = tree.nodes['Render Layers']
    composite = tree.nodes['Composite']
    norm = tree.nodes['Normalize']
    linkNodes(tree, render, norm, out=2)
    linkNodes(tree, norm, composite)
    bpy.data.worlds['World'].horizon_color = (0, 0, 0)
    bpy.data.scenes['Scene'].render.resolution_percentage = 100

setup()

port = 59892
host = "localhost"

logfile = None
quiet = False
rpyc.lib.setup_logger(quiet, logfile)

t = BlockingServer(rpyc.core.SlaveService, hostname = host, port = port,
    reuse_addr = True, ipv6 = False, authenticator = None,
    registrar = None, auto_register = None)
t.start()
