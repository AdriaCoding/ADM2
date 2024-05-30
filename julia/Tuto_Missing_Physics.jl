# SciML Tools
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays
# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)

# Define the Lotka-Volterra equations
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0, 5.0)
u0 = 5.0f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])

rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
              Lux.Dense(5, 2))
# Get the initial parameters and state variables of the model
p_nn, st = Lux.setup(rng, U)

const _st = st
p= ComponentArray(NN=p_nn, LV=rand(rng, Float32,4))

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.NN, _st)[1] # Forward pass
    α, β, γ, δ = p.LV
    # Lokta-Volterra equations + ANN
    du[1] = α*u[1] - β*u[1]*u[2] + û[1]
    du[2] = γ*u[1]*u[2] - δ*u[2] + û[2]
end

# Define the problem
prob_nn = ODEProblem(ude_dynamics!, Xₙ[:, 1], tspan, p)

function predict(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol = 1e-6, reltol = 1e-6,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

@time predict(p)

function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xₙ .- X̂)
end

@time loss(p)

losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

import Base: real
real(p::NamedTuple{T}) where T = 0f0


res1 = Optimization.solve(optprob, ADAM(), callback = callback, maxiters = 1000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 1000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.u

# Plot the losses
pl_losses = plot(1:1000, losses[1:1000], yaxis = :log10, xaxis = :log10,
                 xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(1001:length(losses), losses[1001:end], yaxis = :log10, xaxis = :log10,
      xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)

      ## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):(mean(diff(solution.t)) / 2):last(solution.t)
X̂ = predict(p_trained, Xₙ[:, 1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel = "x(t), y(t)", color = :red,
                     label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])

# See behaviour of out modle in the future
tspan_future = (0.0, 40.0)
prob_future = ODEProblem(lotka!, u0, tspan_future, p_)
solution_f = solve(prob_future, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)
ts_f = first(solution_f.t):(mean(diff(solution_f.t)) / 2):last(solution_f.t)
X̂_f = predict(p_trained, Xₙ[:, 1], ts_f)
pl_trajectory = plot(ts_f, transpose(X̂_f), xlabel = "t", ylabel = "x(t), y(t)", color = :red,
                     label = ["UDE Approximation" nothing])
plot!(solution_f, alpha = 0.75, color = :black, label = ["True Data" nothing])

println("Our real parameters and the ones obtained by the NN are quite unsimilar")
println("Real  -> ", round.(p_, digits=1))
println("Model -> ", round.(p_trained[:LV]', digits=2))