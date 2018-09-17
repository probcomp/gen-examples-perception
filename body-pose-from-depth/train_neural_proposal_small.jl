include("train_neural_proposal.jl")

println("generating training data...")
blender = "blender"
model = "HumanKTH.decimated.blend"
const renderer = BodyPoseDepthRenderer(width, height, blender, model, 59897)
Gen.load_generated_functions()
const training_data = generate_training_data(renderer)

println("small arch...")
arch_small = NetworkArchitecture(8, 8, 16, 128)
proposal_small = make_neural_proposal(arch_small)
session = init_session!(proposal_small.network)
#as_default(GenTF.get_graph(proposal_small.network)) do
    #saver = tf.train.Saver()
    #tf.train.restore(saver, session, "params_small_arch.jld")
#end
Gen.load_generated_functions()
train_inference_network(training_data, 100, 1000000, proposal_small, "params_small_arch_8_8_16_128.jld", session)

close(renderer)
