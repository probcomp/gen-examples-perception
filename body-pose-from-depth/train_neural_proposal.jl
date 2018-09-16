import FileIO
import Random
using Printf
using Gen
using GenTF
import ReverseDiff
using TensorFlow
tf = TensorFlow

include("model.jl")
include("neural_proposal.jl")


const num_train = 100 # 100,000

function generate_training_data(renderer)
    traces = Vector{Any}(num_train)
    for i=1:num_train
        traces[i] = get_choices(simulate(generative_model, (renderer,)))
        if i % 100 == 0
            println("$i of $num_train")
        end
    end
    traces
end

function train_inference_network(training_data, batch_size::Int, num_iter::Int,
                                 proposal::NeuralProposal, params_fname, session::Session)
    for iter=1:num_iter
        minibatch = Random.randperm(num_train)[1:batch_size]
        choices_arr = training_data[minibatch]
        images = Matrix{Float64}[choices[:image] for choices in choices_arr]
        latents = vectorize_internal([get_internal_node(choices, :pose) for choices in choices_arr])
        constraints = DynamicChoiceTrie()
        set_internal_node!(constraints, :poses, latents)
        batched_trace = assess(proposal.neural_proposal_batched, (images,), constraints)
        score = get_call_record(batched_trace).score / batch_size
        println("iter $iter, avg score: $score")
        backprop_params(proposal.neural_proposal_batched, batched_trace, nothing)
        tf.run(session, proposal.network_update)
        if iter % 10 == 0
            as_default(GenTF.get_graph(proposal.network)) do
                saver = tf.train.Saver()
                println("saving params to $params_fname...")
                save(saver, session, params_fname)
            end
        end
    end
end

println("generating training data...")
const renderer = BodyPoseRenderer(width, height, "localhost", 59893)
Gen.load_generated_functions()
const training_data = generate_training_data(renderer)

println("small arch...")
arch_small = NetworkArchitecture(8, 8, 16, 128)
proposal_small = make_neural_proposal(arch_small)
session = init_session!(proposal_small.network)
as_default(GenTF.get_graph(proposal_small.network)) do
    saver = tf.train.Saver()
    tf.train.restore(saver, session, "params_small_arch.jld")
end
Gen.load_generated_functions()
train_inference_network(training_data, 20, 10, proposal_small, "params_small_arch.jld", session)

println("large arch...")
arch_large = NetworkArchitecture(32, 32, 64, 1024)
proposal_large = make_neural_proposal(arch_large)
session = init_session!(proposal_large.network)
as_default(GenTF.get_graph(proposal_large.network)) do
    saver = tf.train.Saver()
    tf.train.restore(saver, session, "params_large_arch.jld")
end
Gen.load_generated_functions()
train_inference_network(training_data, 20, 10, proposal_large, "params_large_arch.jld", session)
