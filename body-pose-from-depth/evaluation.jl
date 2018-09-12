using DataFrames
using Gen

include("scene.jl")
include("renderer.jl")

####################
# generative model #
####################

struct NoisyMatrix <: Gen.Distribution{Matrix{Float64}} end

const noisy_matrix = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64}, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Gen.random(::NoisyMatrix, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = mu[i, j] + randn() * noise
        end
    end
    return mat
end

Gen.get_static_argument_types(::NoisyMatrix) = [Matrix{Float64}, Float64]

# NOTE: this happens to use the same model as the data generator for evaluation

@compiled @gen function generative_model(renderer)
    pose::BodyPose = @addr(body_pose_model(), :pose)
    image::Matrix{Float64} = render(renderer, pose)
    blurred::Matrix{Float64} = imfilter(image, Kernel.gaussian(1))
    observable::Matrix{Float64} = @addr(noisy_matrix(blurred, 0.1), :image)
    return (image, blurred, observable)::Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
end



##############
# evaluation #
##############

abstract type InferenceProgram end

function evaluate(ground_truth, percept, inference_programs::Dict{String, InferenceProgram}, replicates::Int)
    keys = String[]
    square_errors = Float64[]
    elapsed = Float64[]
    for (key, program) in inference_programs
        for rep=1:replicates
            start = time()
            latents = infer(program, percept)
            push!(elapsed, time() - start)
            push!(square_errors, square_error(latents, ground_truth))
            push!(keys, key)
        end
    end
    df = DataFrame()
    df[:elapsed] = elapsed
    df[:square_error] = square_errors
    df[:key] = keys
    add_columns!(ground_truth, df) # adds columns, with the same value for all rows
    return df
end

function evaluate(scene_model, renderer, inference_programs::Dict{String, InferenceProgram}, num_percepts::Int, replicates::Int)
    dfs = Vector{DataFrame}(undef, num_percepts)
    for i=1:num_percepts
        # NOTE: could be modified to use real-world labelled training data
        ground_truth = sample(scene_model)
        percept = render(renderer, ground_truth)
        push!(dfs, evaluate(ground_truth, percept, inference_programs, replicates))
    end
    df = vcat(dfs...)
end


# then we get a data frame where each row is one run of one inference program

# we compute the RMSE and the median runtime

#################
# do experiment #
#################
