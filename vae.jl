using Gen

########################
# the generative model #
########################

function generate_data()
    k = uniform_discrete(1, 3)
    if k == 1
        x = normal(0, 1)
        y = normal(0, 1)
    elseif k == 2
        x = normal(0, 1)
        y = normal(1, 1)
    else
        x = normal(1, 1)
        y = normal(0, 1)
    end
end

# TODO use the mug as a more realistic example?

#####################
# VAE as a proposal #
#####################

# for y given x

function net(a, b, params)
    (W1, b1, W2, b2) = params
    h = W1 * [a, b] .+ b1 # W1 is (K,2) and b1 is (K,)
    W2 * atan.(h) .+ b2 # W2 is (2,K) and b2 is (2,)
end

function decoder_net(z, x::Float64, params)
    net(z, x, params)
end

function encoder_net(x::Float64, y::Float64, params)
    net(x, y, params)
end

@gen function P(x::Float64)
    @param W1::Matrix{Float64}
    @param b1::Matrix{Float64}
    @param W2::Matrix{Float64}
    @param b2::Matrix{Float64}
    params = (W1, b1, W2, b2)
    z = @trace(normal(0, 1), :z)
    y_mu, y_log_std = decoder_net(z, x, params)
    @trace(normal(y_mu, exp(y_log_std)), :y)
    return nothing
end

@gen function Q_net(x::Float64, y::Float64)
    @param W1::Matrix{Float64}
    @param b1::Matrix{Float64}
    @param W2::Matrix{Float64}
    @param b2::Matrix{Float64}
    params = (W1, b1, W2, b2)
    encoder_net(x, y, params)
end

@gen function Q(x::Float64, y::Float64)
    (z_mu, z_log_std) = @trace(Q_net(x, y))
    return @trace(normal(z_mu, exp(z_log_std)), :z)
end

@gen (grad) function Q_reparam(x::Float64, y::Float64)
    (z_mu, z_log_std) = @trace(Q_net(x, y))
    r = @trace(normal(0, 1), :r)
    z = z_mu + (r * exp(z_log_std))
    return z # TODO how to generalize this pattern?
end

function sample_from_Q(x::Float64, y::Float64, q_params)
    (W1, b1, W2, b2) = q_params
    return Q(x, y)
end

function estimate_P_grads(x::Float64, y::Float64, p_params, q_params, N::Int)
    (W1, b1, W2, b2) = p_params
    init_param!(P, :W1, W1)
    init_param!(P, :b1, b1)
    init_param!(P, :W2, W2)
    init_param!(P, :b2, b2)
    (W1, b1, W2, b2) = q_params
    init_param!(Q_net, :W1, W1)
    init_param!(Q_net, :b1, b1)
    init_param!(Q_net, :W2, W2)
    init_param!(Q_net, :b2, b2)
    for i=1:N
        z = Q(x, y)
        tr, = generate(P, (x,), choicemap((:z, z)))
        accumulate_param_gradients!(tr, nothing)
    end
    W1_grad = get_param_grad(P, :W1)
    b1_grad = get_param_grad(P, :b1)
    W2_grad = get_param_grad(P, :W2)
    b2_grad = get_param_grad(P, :b2)
    (W1_grad / N, b1_grad / N, W2_grad / N, b2_grad / N)
end

# TODO how to generalize this pattern?
# including to cases when only some of the choices are reparametrized?

function estimate_Q_grads(x::Float64, y::Float64, p_params, q_params, N::Int)
    (W1, b1, W2, b2) = p_params
    init_param!(P, :W1, W1)
    init_param!(P, :b1, b1)
    init_param!(P, :W2, W2)
    init_param!(P, :b2, b2)
    (W1, b1, W2, b2) = q_params
    init_param!(Q_net, :W1, W1)
    init_param!(Q_net, :b1, b1)
    init_param!(Q_net, :W2, W2)
    init_param!(Q_net, :b2, b2)
    for i=1:N
        # L_Q = log q(z; x, y)
        # L_P = log p(z, y; x)
        # - dL_Q / dphi - (dL_Q / dz * dz / dphi) + (dL_P / dz * dz / d phi)

        # - dL_Q / dphi
        Q_tr = simulate(Q, (x,y))
        accumulate_param_gradients!(Q_tr, nothing, -1) # scale factor -1

        #- (dL_Q / dz * dz / dphi) + (dL_P / dz * dz / d phi) 
        Q_reparam_tr = simulate(Q_reparam, (x,y))
        z = get_retval(Q_reparam_tr)
        P_tr, = generate(P, (x,), choicemap((:z, z), (:y, y)))
        _, _, P_choice_grads = choice_gradients(P_tr, select(:z), nothing)
        P_z_grad = P_choice_grads[:z] # d log p(z, y; x) / dz
        Q_tr, = generate(Q, (x,y), choicemap((:z, z)))
        _, _, Q_choice_grads = choice_gradients(Q_tr, select(:z), nothing)
        Q_z_grad = Q_choice_grads[:z] # d log p(z, y; x) / dz
        z_grad_diff = P_z_grad - Q_z_grad
        accumulate_param_gradients!(Q_reparam_tr, z_grad_diff) # scale factor +1

    end
    W1_grad = get_param_grad(Q_net, :W1)
    b1_grad = get_param_grad(Q_net, :b1)
    W2_grad = get_param_grad(Q_net, :W2)
    b2_grad = get_param_grad(Q_net, :b2)
    (W1_grad / N, b1_grad / N, W2_grad / N, b2_grad / N)
end

const M = 0.0001

function generate_init_params(K::Int)
    W1 = randn(K,2) * M
    b1 = zeros(K)
    W2 = randn(2,K) * M
    b2 = zeros(2)
    (W1, b1, W2, b2)
end

p_params = generate_init_params(10)
q_params = generate_init_params(10)

p_grads = estimate_P_grads(1., 2., p_params, q_params, 100)
q_grads = estimate_Q_grads(1., 2., p_params, q_params, 100)

println(p_grads)
println(q_grads)


# TODO the gradient wrt params of P is just obtained by sampling from Q once, and taking the gradient of the joint probability with respect to the params..
# the gradient with respect to the params of Q is obtained (with the reparam trick) by 
# (i) sampling z once, then taking the gradient respect to the parameters, 

#struct VAETrace
    #gen_fn::GenerativeFunction
    #args::Any
    #score::Float64
    #output_addr::Any
    #output::Any
#end
#
#struct VAE <: GenerativeFunction{Any, VAETrace}
#end



# need to implement:
# generate()
#arg_grads = accumulate_param_gradients!(trace, retgrad, scale_factor=1.)
# propose()

# show that it captures multimodality (contrast against a simpler model that
# just parametrizes a Gaussian)

# reparametrization trick..
