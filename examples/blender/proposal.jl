

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


inference_network = @tf_function begin

    const num_conv1 = 32
    const num_conv2 = 32
    const num_conv3 = 64
    const num_fc = 1024
    const num_output = 32

    # input image
    @input image_flat Float32 [-1, num_pixels]
    image = tf.reshape(image_flat, [-1, width, height, 1])

    # convolution + max-pooling
    @param W_conv1 initial_weight([5, 5, 1, num_conv1])
    @param b_conv1 initial_bias([num_conv1])
    h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # convolution + max-pooling
    @param W_conv2 initial_weight([5, 5, num_conv1, num_conv2])
    @param b_conv2 initial_bias([num_conv2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, div(width, 4) * div(height, 4) * num_conv2])

    # convolution + max-pooling
    @param W_conv3 initial_weight([5, 5, num_conv2, num_conv3])
    @param b_conv3 initial_bias([num_conv3])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_pool3_flat = tf.reshape(h_pool3, [-1, div(width, 8) * div(height, 8) * num_conv3])

    # fully connected layer
    @param W_fc1 initial_weight([div(width, 8) * div(height, 8) * num_conv3, num_fc])
    @param b_fc1 initial_bias([num_fc])
    h_fc1 = tf.nn.relu(h_pool3_flat * W_fc1 + b_fc1)

    # output layer
    @param W_fc2 initial_weight([num_fc, num_output])
    @param b_fc2 initial_bias([num_output])
    @output Float32 (tf.matmul(h_fc1, W_fc2) + b_fc2)
end

function make_inference_network_update()
    net = inference_network

    # get accumulated negative gradients of log probability with respect to each parameter
    grads_and_vars = [
        (tf.negative(get_param_grad(net, n)), get_param_val(net, n)) for n in get_param_names(net)]

    # use ADAM 
    optimizer = tf.train.AdamOptimizer(1e-4)

    tf.group(
        tf.train.apply_gradients(optimizer, grads_and_vars),
        [zero_grad(net, n) for n in get_param_names(net)]...)
end

inference_network_update = make_inference_network_update()

dl_proposal_predict = @generative function (outputs)

    # TODO capture dependencies within e.g. right elbow using multivariate e.g.
    # gaussian proposals (use Cholesky decomposition, parametrize by L) see
    # Unconstrained Parameterizations for Variance-Covariance Matrices
    # (Pinheiro et al.)

    i = 1
    
    # whole body rotation
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "rotation"); i += 2

    # right elbow
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "arm_elbow_r_location_dx"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "arm_elbow_r_location_dy"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "arm_elbow_r_location_dz"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "arm_elbow_r_rotation_dz"); i += 2

    # left elbow
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "arm_elbow_l_location_dx"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "arm_elbow_l_location_dy"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "arm_elbow_l_location_dz"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "arm_elbow_l_rotation_dz"); i += 2

    # hip
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "hip_location_dz"); i += 2

    # right heel
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "heel_r_location_dx"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "heel_r_location_dy"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "heel_r_location_dz"); i += 2

    # left heel
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "heel_l_location_dx"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "heel_l_location_dy"); i += 2
    @rand(Gen.beta(exp(outputs[i]), exp(outputs[i+1])), "heel_l_location_dz"); i += 2
end

dl_proposal = @generative function ()

    # get image from input trace
    image_flat = zeros(1, num_pixels)
    image_flat[1,:] = @read("image")[:]

    # run inference network
    outputs = @tf_call(inference_network(image_flat))

    # make prediction given inference network outputs
    @splice(dl_proposal_predict(outputs[1,:]))
end

dl_proposal_batched = @generative function (batch_size::Int)

    # get images from input trace
    images_flat = zeros(Float32, batch_size, width * height)
    for i=1:batch_size
        images_flat[i,:] = @read(("$i", "image"))[:]
    end

    # run inference network in batch
    outputs = @tf_call(inference_network(images_flat))
    
    # make prediction for each image given inference network outputs
    for i=1:batch_size
        @call(dl_proposal_predict(outputs[i,:]), "$i")
    end
end
