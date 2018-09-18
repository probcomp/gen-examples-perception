include("train_neural_proposal.jl")

println("generating training data...")
blender = "blender"
model = "HumanKTH.decimated.blend"
const renderer = BodyPoseDepthRenderer(width, height, blender, model, 59898)
Gen.load_generated_functions()
const training_data = generate_training_data(renderer, 100000)

arch = NetworkArchitecture(8, 8, 16, 128)
proposal = make_neural_proposal(arch)
session = init_session!(proposal.network)
params_fname = "params_arch_8_8_16_128.jld"
#as_default(GenTF.get_graph(proposal.network)) do
    #saver = tf.train.Saver()
    #tf.train.restore(saver, session, params_fname)
#end
Gen.load_generated_functions()
train_inference_network(training_data, 100, 1000000, proposal, params_fname, session)

close(renderer)
