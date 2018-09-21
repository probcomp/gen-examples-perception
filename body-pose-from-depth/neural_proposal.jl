import GenTF
import TensorFlow
tf = TensorFlow
import ReverseDiff
using Printf
import Random

using FileIO
using GenTF: @tf_function, @input, @param, @output
using GenTF: get_graph, TensorFlowFunction, init_session!
using GenTF: get_param_names, get_param_grad, get_param_val, zero_grad
using TensorFlow: as_default, Tensor, Session, as_default


function conv2d(x, W)
    tf.nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
end

function max_pool_2x2(x)
    tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end

function initial_weight(shape)
    randn(Float32, shape...) * 0.001f0
end

function initial_bias(shape)
    fill(0.1f0, shape...)
end

const num_output = 32

struct NetworkArchitecture
    num_conv1::Int
    num_conv2::Int
    num_conv3::Int
    num_fc::Int
end

function make_inference_network(arch::NetworkArchitecture)

    @tf_function begin

        # input image
        @input image_flat Float32 [-1, width * height]
        image = tf.reshape(image_flat, [-1, width, height, 1])
    
        # convolution + max-pooling
        @param W_conv1 initial_weight([5, 5, 1, arch.num_conv1])
        @param b_conv1 initial_bias([arch.num_conv1])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
    
        # convolution + max-pooling
        @param W_conv2 initial_weight([5, 5, arch.num_conv1, arch.num_conv2])
        @param b_conv2 initial_bias([arch.num_conv2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, div(width, 4) * div(height, 4) * arch.num_conv2])
    
        # convolution + max-pooling
        @param W_conv3 initial_weight([5, 5, arch.num_conv2, arch.num_conv3])
        @param b_conv3 initial_bias([arch.num_conv3])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
        h_pool3_flat = tf.reshape(h_pool3, [-1, div(width, 8) * div(height, 8) * arch.num_conv3])
    
        # fully connected layer
        @param W_fc1 initial_weight([div(width, 8) * div(height, 8) * arch.num_conv3, arch.num_fc])
        @param b_fc1 initial_bias([arch.num_fc])
        h_fc1 = tf.nn.relu(h_pool3_flat * W_fc1 + b_fc1)
    
        # output layer
        @param W_fc2 initial_weight([arch.num_fc, num_output])
        @param b_fc2 initial_bias([num_output])
        @output Float32 (tf.matmul(h_fc1, W_fc2) + b_fc2)
    end
end

function make_update(net::TensorFlowFunction)
    as_default(GenTF.get_graph(net)) do 

        # get accumulated negative gradients of log probability with respect to each parameter
        grads_and_vars = [
            (tf.negative(get_param_grad(net, n)), get_param_val(net, n)) for n in get_param_names(net)]
        # do ADAM step and then reset the gradient accumulators
        optimizer = tf.train.AdamOptimizer(1e-4)
        step = tf.train.apply_gradients(optimizer, grads_and_vars)
        tf.with_op_control([step]) do
            tf.group([zero_grad(net, n) for n in get_param_names(net)]...)
        end
    end
end

#@compiled @gen function neural_proposal_predict(@ad(outputs::Vector{Float64}))
@gen function neural_proposal_predict_beta(@ad(outputs::Vector{Float64}))

    # global rotation
    @addr(beta(exp(outputs[1]), exp(outputs[2])), :rot_z)

    # right elbow location
    @addr(beta(exp(outputs[3]), exp(outputs[4])), :elbow_r_loc_x)
    @addr(beta(exp(outputs[5]), exp(outputs[6])), :elbow_r_loc_y)
    @addr(beta(exp(outputs[7]), exp(outputs[8])), :elbow_r_loc_z)

    # left elbow location
    @addr(beta(exp(outputs[11]), exp(outputs[12])), :elbow_l_loc_x)
    @addr(beta(exp(outputs[13]), exp(outputs[14])), :elbow_l_loc_y)
    @addr(beta(exp(outputs[15]), exp(outputs[16])), :elbow_l_loc_z)

    # right elbow rotation
    @addr(beta(exp(outputs[9]), exp(outputs[10])), :elbow_r_rot_z)

    # left elbow rotation
    @addr(beta(exp(outputs[17]), exp(outputs[18])), :elbow_l_rot_z)

    # hip
    @addr(beta(exp(outputs[19]), exp(outputs[20])), :hip_loc_z)

    # right heel
    @addr(beta(exp(outputs[21]), exp(outputs[22])), :heel_r_loc_x)
    @addr(beta(exp(outputs[23]), exp(outputs[24])), :heel_r_loc_y)
    @addr(beta(exp(outputs[25]), exp(outputs[26])), :heel_r_loc_z)

    # left heel
    @addr(beta(exp(outputs[27]), exp(outputs[28])), :heel_l_loc_x)
    @addr(beta(exp(outputs[29]), exp(outputs[30])), :heel_l_loc_y)
    @addr(beta(exp(outputs[31]), exp(outputs[32])), :heel_l_loc_z)
end

@gen function neural_proposal_predict_normal(@ad(outputs::Vector{Float64}))

    # global rotation
    @addr(normal(outputs[1], exp(outputs[2])), :rot_z)

    # right elbow location
    @addr(normal(outputs[3], exp(outputs[4])), :elbow_r_loc_x)
    @addr(normal(outputs[5], exp(outputs[6])), :elbow_r_loc_y)
    @addr(normal(outputs[7], exp(outputs[8])), :elbow_r_loc_z)

    # left elbow location
    @addr(normal(outputs[11], exp(outputs[12])), :elbow_l_loc_x)
    @addr(normal(outputs[13], exp(outputs[14])), :elbow_l_loc_y)
    @addr(normal(outputs[15], exp(outputs[16])), :elbow_l_loc_z)

    # right elbow rotation
    @addr(normal(outputs[9], exp(outputs[10])), :elbow_r_rot_z)

    # left elbow rotation
    @addr(normal(outputs[17], exp(outputs[18])), :elbow_l_rot_z)

    # hip
    @addr(normal(outputs[19], exp(outputs[20])), :hip_loc_z)

    # right heel
    @addr(normal(outputs[21], exp(outputs[22])), :heel_r_loc_x)
    @addr(normal(outputs[23], exp(outputs[24])), :heel_r_loc_y)
    @addr(normal(outputs[25], exp(outputs[26])), :heel_r_loc_z)

    # left heel
    @addr(normal(outputs[27], exp(outputs[28])), :heel_l_loc_x)
    @addr(normal(outputs[29], exp(outputs[30])), :heel_l_loc_y)
    @addr(normal(outputs[31], exp(outputs[32])), :heel_l_loc_z)
end

struct NeuralProposal
    arch::NetworkArchitecture
    network::TensorFlowFunction
    network_update::Tensor
    neural_proposal::Generator
    neural_proposal_batched::Generator
end

function make_neural_proposal(arch::NetworkArchitecture, predictor::Gen.GenFunction)
    network = make_inference_network(arch)
    update = make_update(network)

    neural_proposal = gensym("neural_proposal")
    eval(quote
        @compiled @gen function $neural_proposal(@ad(image::Matrix{Float64}))

            # run inference network
            image_flat::Matrix{Float64} = reshape(image, 1, width * height)
            outputs::Matrix{Float64} = @addr($(QuoteNode(network))(image_flat), :network)

            # make prediction given inference network outputs
            @addr($(QuoteNode(predictor))(outputs[1,:]), :pose)
        end
    end)

    neural_proposal_batched = gensym("neural_proposal_batched")
    eval(quote
        @gen function $neural_proposal_batched(images::Vector{Matrix{Float64}})

            # get images from input trace
            batch_size = length(images)
            println("batch size: $batch_size")
            images_flat = zeros(Float32, batch_size, width * height)
            for i=1:batch_size
                images_flat[i,:] = images[i][:]
            end
            println("size(images_flat): $(size(images_flat))")
        
            # run inference network in batch
            outputs = @addr($(QuoteNode(network))(images_flat), :network)
            println("size(outputs): $(size(outputs))")
            
            # make prediction for each image given inference network outputs
            for i=1:batch_size
                @addr($(QuoteNode(predictor))(outputs[i,:]), :poses => i)
            end
        end
    end)
    
    NeuralProposal(arch, network, update,
        eval(neural_proposal), eval(neural_proposal_batched))
end


function train_inference_network(num_batch::Int, batch_size::Int,
                                 num_minibatch::Int, minibatch_size::Int, 
                                 proposal::NeuralProposal, params_fname_trunk,
                                 session::Session, renderer; verbose=false)

    function input_extractor(teacher_choices_arr::Vector{Any})
        @assert length(teacher_choices_arr) == minibatch_size
        images = Matrix{Float64}[choices[:image] for choices in teacher_choices_arr]
        (images,)
    end

    function constraint_extractor(teacher_choices_arr::Vector{Any})
        @assert length(teacher_choices_arr) == minibatch_size
        poses = vectorize_internal([get_internal_node(c, :pose) for c in teacher_choices_arr])
        constraints = DynamicChoiceTrie()
        set_internal_node!(constraints, :poses, poses)
        constraints
    end

    function minibatch_callback(batch::Int, minibatch::Int, avg_score::Float64, verbose::Bool)
        if verbose
            println("batch $batch of $num_batch, minibatch $minibatch of $num_minibatch, avg score: $avg_score")
        end
        tf.run(session, proposal.network_update)
    end
    
    function batch_callback(batch::Int, verbose::Bool)
        as_default(GenTF.get_graph(proposal.network)) do
            saver = tf.train.Saver()
            params_fname = "$(params_fname_trunk)-$batch.jld"
            println("finished batch $batch, saving params to $params_fname...")
            save(saver, session, params_fname)
        end
    end

    conf = SGDTrainConf(
                num_batch,
                batch_size,
                num_minibatch,
                minibatch_size,
                input_extractor,
                constraint_extractor,
                minibatch_callback,
                batch_callback)
    sgd_train_batch(generative_model, (renderer,), proposal.neural_proposal_batched, conf, verbose)
end

