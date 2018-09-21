using Gen
using Test
using PyPlot
using Random: seed!

include("spike_and_slab.jl")

###############
# test logpdf #
###############

spike_mass = 0.7
slab_mass = 1. - spike_mass

# case 1: overhang on left
center = 0.2
width = 0.6
nominal_spike_density = spike_mass / width
expected = log(2 * spike_mass * nominal_spike_density + slab_mass)
@test isapprox(logpdf(spike_slab, 0.05, center, width, spike_mass), expected)
expected = log(spike_mass * nominal_spike_density + slab_mass)
@test isapprox(logpdf(spike_slab, 0.15, center, width, spike_mass), expected)
expected = log(slab_mass)
@test isapprox(logpdf(spike_slab, 0.9, center, width, spike_mass), expected)

# case 2: overhang on right
center = 0.8
width = 0.6
nominal_spike_density = spike_mass / width
expected = log(2 * spike_mass * nominal_spike_density + slab_mass)
@test isapprox(logpdf(spike_slab, 0.95, center, width, spike_mass), expected)
expected = log(spike_mass * nominal_spike_density + slab_mass)
@test isapprox(logpdf(spike_slab, 0.85, center, width, spike_mass), expected)
expected = log(slab_mass)
@test isapprox(logpdf(spike_slab, 0.1, center, width, spike_mass), expected)

# case 3: no overhang
center = 0.6
width = 0.3
nominal_spike_density = spike_mass / width
expected = log(slab_mass)
@test isapprox(logpdf(spike_slab, 0.1, center, width, spike_mass), expected)
expected = log(spike_mass * nominal_spike_density + slab_mass)
@test isapprox(logpdf(spike_slab, 0.5, center, width, spike_mass), expected)
expected = log(slab_mass)
@test isapprox(logpdf(spike_slab, 0.9, center, width, spike_mass), expected)

###############
# test random #
###############

figure(figsize=(8, 6))

seed!(0)
n = 100000
spike_mass = 0.7
slab_mass = 0.3

# case 1: overhang on left
center = 0.2
width = 0.6
overhang = 0.1
xs = Float64[spike_slab(center, width, spike_mass) for _=1:n]
num_slab = sum(xs .> center + width / 2.)
frac_slab = num_slab / n
@test isapprox(frac_slab, slab_mass * (1. - (center + width/2.)), atol=1e-2)
subplot(2, 3, 1)
title("center: $center, with: $width, mass: $spike_mass")
PyPlot.plt[:hist](xs, bins=30)
subplot(2, 3, 4)
title("center: $center, with: $width, mass: $spike_mass")
probes = range(0., stop=1., length=100)
pdf = map((x) -> exp(logpdf(spike_slab, x, center, width, spike_mass)), probes)
PyPlot.plot(probes, pdf)
gca()[:set_ylim](0, 2)

# case 2: overhang on right
center = 0.8
width = 0.6
overhang = 0.1
xs = Float64[spike_slab(center, width, spike_mass) for _=1:n]
num_slab = sum(xs .< center - width / 2.)
frac_slab = num_slab / n
@test isapprox(frac_slab, slab_mass * (center - width/2.), atol=1e-2)
subplot(2, 3, 2)
title("center: $center, with: $width, mass: $spike_mass")
PyPlot.plt[:hist](xs, bins=30)
subplot(2, 3, 5)
title("center: $center, with: $width, mass: $spike_mass")
probes = range(0., stop=1., length=100)
pdf = map((x) -> exp(logpdf(spike_slab, x, center, width, spike_mass)), probes)
PyPlot.plot(probes, pdf)
gca()[:set_ylim](0, 2)


# case 3: no overhang
center = 0.6
width = 0.3
xs = Float64[spike_slab(center, width, spike_mass) for _=1:n]
num_slab = sum((xs .< center - width / 2.) .| (xs .> center + width / 2.))
frac_slab = num_slab / n
@test isapprox(frac_slab, slab_mass * (1. - width), atol=1e-2)
subplot(2, 3, 3)
title("center: $center, with: $width, mass: $spike_mass")
PyPlot.plt[:hist](xs, bins=30)
subplot(2, 3, 6)
title("center: $center, with: $width, mass: $spike_mass")
probes = range(0., stop=1., length=100)
pdf = map((x) -> exp(logpdf(spike_slab, x, center, width, spike_mass)), probes)
PyPlot.plot(probes, pdf)
gca()[:set_ylim](0, 2)


savefig("spike_and_slab_hist.png")


