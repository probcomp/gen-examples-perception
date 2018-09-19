include("model.jl")
include("neural_proposal.jl")

port = parse(Int, ARGS[1])
println("port: $port")

blender = "blender"
model = "HumanKTH.decimated.blend"
const renderer = BodyPoseDepthRenderer(width, height, blender, model, port)

arch = NetworkArchitecture(4, 4, 8, 32)
proposal = make_neural_proposal(arch)
session = init_session!(proposal.network)
params_fname_trunk = "/data/params_arch_4_4_8_32-$port"
#as_default(GenTF.get_graph(proposal.network)) do
    #saver = tf.train.Saver()
    #tf.train.restore(saver, session, params_fname)
#end
Gen.load_generated_functions()

num_batch = 100000 # forever
batch_size = 10000
num_minibatch = 300
minibatch_size = 100
train_inference_network(
        num_batch, batch_size, num_minibatch, minibatch_size,
        proposal, params_fname_trunk, session, renderer; verbose=true)

close(renderer)
