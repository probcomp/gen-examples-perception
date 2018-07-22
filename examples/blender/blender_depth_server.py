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
   
class BlenderService(rpyc.Service):

    def __init__(self):
        super().__init__()
        self.scene = bpy.data.scenes['Scene']

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_open_mainfile(self, filepath):
        bpy.ops.wm.open_mainfile(filepath=filepath)

    def exposed_setup_for_depth(self):
        self.scene.use_nodes = True
        self.scene.render.layers[0].use_pass_normal = True # ?
        tree = bpy.context.scene.node_tree
        addNode(tree, 'CompositorNodeNormalize')
        render = tree.nodes['Render Layers']
        composite = tree.nodes['Composite']
        norm = tree.nodes['Normalize']
        linkNodes(tree, render, norm, out=2)
        linkNodes(tree, norm, composite)
        bpy.data.worlds['World'].horizon_color = (0, 0, 0)
        self.scene.render.resolution_percentage = 100

    def exposed_set_resolution(self, x, y):
        self.scene.render.resolution_x = x
        self.scene.render.resolution_y = y
        self.scene.render.resolution_percentage = 100

    def exposed_add_plane(self, name, location, rotation_euler, scale):
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.object
        obj.location = location
        obj.rotation_euler = rotation_euler
        obj.scale = scale
        obj.name = name
   
    def exposed_set_object_location(self, name, location):
        obj = bpy.data.objects[name]
        obj.location = location

    def exposed_set_object_rotation_euler(self, name, rotation_euler):
        obj = bpy.data.objects[name]
        obj.rotation_euler = rotation_euler

    def exposed_set_object_scale(self, name, scale):
        obj = bpy.data.objects[name]
        obj.scale = scale

    def exposed_render(self, filepath):
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

t = BlockingServer(BlenderService, port=59892)
t.start()
