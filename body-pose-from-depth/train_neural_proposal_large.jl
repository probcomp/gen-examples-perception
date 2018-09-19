include("train_neural_proposal.jl")

blender = "blender"
model = "HumanKTH.decimated.blend"
const renderer = BodyPoseDepthRenderer(width, height, blender, model, 59899)

arch = NetworkArchitecture(32, 32, 64, 1024)
proposal = make_neural_proposal(arch)
session = init_session!(proposal.network)
params_fname = "params_arch_32_32_64_128.jld"
#as_default(GenTF.get_graph(proposal.network)) do
    #saver = tf.train.Saver()
    #tf.train.restore(saver, session, params_fname)
#end
Gen.load_generated_functions()

num_batch = 100000 # forever
batch_size = 100
num_minibatch = 500
minibatch_size = 100
train_inference_network(num_batch, batch_size, num_minibatch, minibatch_size, proposal, params_fname, session, renderer)

close(renderer)

