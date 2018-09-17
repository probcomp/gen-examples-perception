include("train_neural_proposal.jl")

println("generating training data...")
blender = "blender"
model = "HumanKTH.decimated.blend"
const renderer = BodyPoseDepthRenderer(width, height, blender, model, 59898)
Gen.load_generated_functions()
const training_data = generate_training_data(renderer)

println("large arch...")
arch_large = NetworkArchitecture(32, 32, 64, 1024)
proposal_large = make_neural_proposal(arch_large)
session = init_session!(proposal_large.network)
#as_default(GenTF.get_graph(proposal_large.network)) do
    #saver = tf.train.Saver()
    #tf.train.restore(saver, session, "params_large_arch.jld")
#end
Gen.load_generated_functions()
train_inference_network(training_data, 100, 100000, proposal_large, "params_large_arch_32_32_64_128.jld", session)

close(renderer)

