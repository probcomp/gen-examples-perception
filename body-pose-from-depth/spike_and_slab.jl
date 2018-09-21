using Gen

struct SpikeSlab <: Distribution{Float64} end

const spike_slab = SpikeSlab()

function Gen.logpdf(::SpikeSlab, x::Real, center::Real, width::Real, spike_mass::Real)
    @assert spike_mass > 0 && spike_mass < 1
    @assert center > 0 && center < 1
    @assert width > 0 && width < 1
    slab_mass = 1. - spike_mass
    spike_left = center - width / 2.
    spike_right = center + width / 2.
    nominal_spike_density = spike_mass / width
    if spike_right > 1
        # overhang on right
        extra = spike_right - 1
        nominal_spike_right = 1. - extra
        if x < spike_left
            # in the slab
            log(slab_mass * 1.)
        elseif x < nominal_spike_right
            # in the spike
            log(slab_mass * 1. + spike_mass * nominal_spike_density)
        else
            # in the double spike
            log(slab_mass * 1. + 2 * spike_mass * nominal_spike_density)
        end
    elseif spike_left < 0
        # overhang on left
        extra = -spike_left
        nominal_spike_left = extra
        if x > spike_right
            # in the slab
            log(slab_mass * 1.)
        elseif x > nominal_spike_left
            # in the spike
            log(slab_mass * 1. + spike_mass * nominal_spike_density)
        else
            # in the double spike
            log(slab_mass * 1. + 2 * spike_mass * nominal_spike_density)
        end
    else
        # no overhang
        if x < spike_right && x > spike_left
            # in the spike
            log(spike_mass * nominal_spike_density + slab_mass * 1.)
        else
            # in the slab
            log(slab_mass * 1.)
        end
    end
end

function Gen.random(::SpikeSlab, center::Real, width::Real, spike_mass::Real)
    @assert spike_mass > 0 && spike_mass < 1
    @assert center > 0 && center < 1
    @assert width > 0 && width < 1
    if bernoulli(spike_mass)
        spike_left = center - width / 2.
        spike_right = center + width / 2.
        x = uniform_continuous(spike_left, spike_right)
        if x < 0
            # overhang left
            -x
        elseif x > 1
            # overhang right
            1. - (x - 1)
        else
            x
        end
    else
        uniform_continuous(0, 1)
    end
end

function logpdf_grad(::SpikeSlab, x::Real, center::Real, width::Real, spike_mass::Real)
    # TODO could calculate the actual gradient for width and spike_mass, which is nonzero
    (0., 0., nothing, nothing)
end

(::SpikeSlab)(center, width, spike_mass) = random(SpikeSlab(), center, width, spike_mass)

Gen.has_output_grad(::SpikeSlab) = true
Gen.has_argument_grads(::SpikeSlab) = (true, false, false)
Gen.get_static_argument_types(::SpikeSlab) = [Float64, Float64, Float64]

export spike_slab
