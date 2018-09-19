include("model.jl")
include("neural_proposal.jl")

blender = "blender"
model = "HumanKTH.decimated.blend"
const renderer = BodyPoseDepthRenderer(width, height, blender, model, 59898)

arch = NetworkArchitecture(8, 8, 16, 128)
proposal = make_neural_proposal(arch)
session = init_session!(proposal.network)
params_fname = "params_arch_8_8_16_128.jld"
#as_default(GenTF.get_graph(proposal.network)) do
    #saver = tf.train.Saver()
    #tf.train.restore(saver, session, params_fname)
#end
Gen.load_generated_functions()

num_batch = 100000 # forever
batch_size = 10000
num_minibatch = 500
minibatch_size = 100
train_inference_network(num_batch, batch_size, num_minibatch, minibatch_size, proposal, params_fname, session, renderer; verbose=true)

close(renderer)
