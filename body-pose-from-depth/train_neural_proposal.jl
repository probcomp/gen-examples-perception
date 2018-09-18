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

function generate_training_data(renderer, num_train)
    traces = Vector{Any}(num_train)
    for i=1:num_train
        traces[i] = get_choices(simulate(generative_model, (renderer,)))
        if i % 100 == 0
            println("$i of $num_train")
        end
    end
    traces
end

function train_inference_network(num_batch::Int, batch_size::Int, num_minibatch::Int, minibatch_size::Int, 
                                 proposal::NeuralProposal, params_fname, session::Session, renderer)

    for batch=1:num_batch
        training_data = generate_training_data(renderer, batch_size)
        for iter=1:num_minibatch
            minibatch = Random.randperm(batch_size)[1:minibatch_size]
            choices_arr = training_data[minibatch]
            images = Matrix{Float64}[choices[:image] for choices in choices_arr]
            latents = vectorize_internal([get_internal_node(choices, :pose) for choices in choices_arr])
            constraints = DynamicChoiceTrie()
            set_internal_node!(constraints, :poses, latents)
            batched_trace = assess(proposal.neural_proposal_batched, (images,), constraints)
            score = get_call_record(batched_trace).score / minibatch_size
            println("batch $batch of $num_batch, minibatch $iter of $num_minibatch, avg score: $score")
            backprop_params(proposal.neural_proposal_batched, batched_trace, nothing)
            tf.run(session, proposal.network_update)
        end
        as_default(GenTF.get_graph(proposal.network)) do
            saver = tf.train.Saver()
            println("finished batch $batch, saving params to $params_fname...")
            save(saver, session, params_fname)
        end
    end
end
