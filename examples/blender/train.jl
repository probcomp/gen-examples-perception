import JLD

using GenLiteTF
using TensorFlow
tf = TensorFlow

include("model.jl")
include("proposal.jl")

const num_train = 100000

function generate_training_data()
    traces = Vector{Trace}(num_train)
    for i=1:num_train
        (traces[i], _, _) = simulate(model, ())
        @assert has_value(traces[i], "rotation")
        if i % 100 == 0
            println("$i of $num_train")
        end
    end
    traces
end

function initial_weight(shape)
    randn(Float32, shape...) * 0.001
end

function initial_bias(shape)
    fill(0.1f0, shape...)
end

function train_inference_network(all_traces, num_iter)

    # do training
    batch_size = 100
    tic()
    for iter=1:num_iter
        minibatch = randperm(num_train)[1:batch_size]
        traces = all_traces[minibatch]
        @assert length(traces) == batch_size
        vector_trace = vectorize(traces)
        @assert has_subtrace(vector_trace, "1")
        for i=1:batch_size
            @assert has_value(vector_trace["$i"], "rotation")
        end
        (total_score, _) = backprop(dl_proposal_batched, (batch_size,), vector_trace, vector_trace)
        tf.run(get_ambient_tf_session(), inference_network_update)
        score = total_score / batch_size
        println("iter: $iter, score: $(score)")
        if iter % 100 == 0
            saver = tf.train.Saver()
            println("saving params...")
            tf.train.save(saver, get_ambient_tf_session(), "inference_network_params.jld")
        end
    end
    toc()
end

println("generating training data...")
tic()
const traces = generate_training_data()
toc()

tf.run(get_ambient_tf_session(), tf.global_variables_initializer())

#saver = tf.train.Saver()
#tf.train.restore(saver, get_ambient_tf_session(), "params.jld")

println("training for compilation...")
tic()
train_inference_network(traces, 10)
toc()

println("training...")
tic()
train_inference_network(traces, 1000000)
toc()
