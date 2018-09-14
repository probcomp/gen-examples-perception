import JLD2
import FileIO
import Random
using Printf
using Gen
using GenTF
using TensorFlow
tf = TensorFlow

include("model.jl")
include("neural_proposal.jl")

Gen.load_generated_functions()

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

function train_inference_network(all_choices, num_iter)

    # do training
    batch_size = 100
    for iter=1:num_iter
        minibatch = Random.randperm(num_train)[1:batch_size]
        choices_arr = all_choices[minibatch]
        images = Matrix{Float64}[choices[:image] for choices in choices_arr]
        latents = vectorize_internal([get_internal_node(choices, :pose) for choices in choices_arr])
        constraints = DynamicChoiceTrie()
        set_internal_node!(constraints, :poses, latents)
        print(constraints)
        @assert length(images) == batch_size
        batched_trace = assess(dl_proposal_batched, (images,), constraints)
        score = get_call_record(batched_trace).score / batch_size
        backprop_params(dl_proposal_batched, batched_trace, nothing)
        tf.run(session, inference_network_update)
        println("iter: $iter, score: $(score)")
        if iter % 10 == 0
            saver = tf.train.Saver()
            println("saving params...")
            tf.train.save(saver, session, "infer_net_params.jld2")
        end
    end
end

println("generating training data...")
const renderer = BodyPoseRenderer(width, height, "localhost", 59893)
const all_choices = generate_training_data(renderer)

session = Session(get_def_graph())
GenTF.get_session() = session
tf.run(session, tf.global_variables_initializer())

#saver = tf.train.Saver()
#tf.train.restore(saver, get_ambient_tf_session(), "params.jld2")
println("training...")
train_inference_network(all_choices, 10)
