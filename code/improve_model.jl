using PyCall
using Flux

"""
    Catan Game Wrappers
"""
# pushfirst!(pyimport("sys")."path", "/home/polfr/Documents/Stanford/stanford/AA228/Catan-AI/code")

# py"""
# import sys

# def setup():
#     print("test")
#     sys.path.insert(0, "~/home/polfr/Documents/Stanford/stanford/AA228/Catan-AI/code")
# """

# py"setup"()
# function_name = pyimport("AIGame_Wrapper")["play_game_with_policy"]

"""
    Gradient Q Learning
"""

struct GradientQLearning
    A 
    gamma 
    Q
    gradQ
    theta 
    alpha 
end

function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.theta, s, a)
end

scale_gradient(gradient, L2_max) = min(L2_max/norm(gradient), 1) * gradient

function update!(model::GradientQLearning, s, a, r, sp)
    return update!(model::GradientQLearning, s, a, r, sp, [], false)
end

function update!(model::GradientQLearning, s, a, r, sp, usable_actions)
    return update!(model::GradientQLearning, s, a, r, sp, usable_actions, false)
end

function update!(model::GradientQLearning, s, a, r, sp, ignore_expected_util)
    return update!(model::GradientQLearning, s, a, r, sp, [], ignore_expected_util)
end

function update!(model::GradientQLearning, s, a, r, sp, usable_actions, ignore_expected_util)
    A, gamma, Q, theta, alpha = model.A, model.gamma, model.Q, model.theta, model.alpha
    myGrad = model.gradQ(theta, s, a)

    if ignore_expected_util
        u = 0
    elseif isempty(usable_actions)
        u = maximum(Q(theta,sp,ap) for ap in A)
    else
        expected_utility(theta, sp, ap) = ap in usable_actions ? Q(theta,sp,ap) : -Inf
        u = maximum(expected_utility(theta, sp, ap) for ap in A)
    end
    
    for i in range(1,6)
        gradient = (r + gamma*u - Q(theta, s, a)) .* myGrad[theta[i]]
        theta[i] .+= alpha*scale_gradient(gradient, 1)
    end
    return model
end

"""
    Basis function
"""

function Q_func(basis, theta, s, a)
    # params(basis) = theta
    input = vcat(s, a)
    return sum(basis(input))
end

function gradQ_func(basis, theta, s, a)
    return gradient(()->sum(Q_func(basis,theta,s,a)), theta)
end

"""
    Epsilon-Greedy Exploration
"""

mutable struct EpsilonGreedyExploration
    epsilon # probability of random arm
end

function (policy::EpsilonGreedyExploration)(model, s, usable_actions)
    println("here")
    A, epsilon = model.A, π.epsilon
    if rand() < epsilon
        return rand(usable_actions)
    end
    Q(s,a) = lookahead(model, s, a)
    return argmax(a->Q(s,a), usable_actions)
end

"""
    Main function
"""

function main()
    # TODO: Load in the params for theta

    # Basis function
    basis = Chain(
        Dense(57, 32, Flux.relu),
        Dense(32, 16, Flux.relu),
        Dense(16, 1),
    )

    # Gradient Q-Learning
    # A = [[i, j] for j = 1:72 for i=1:54]
    A = 1:54
    gamma = 0.99
    theta = Flux.params(basis)
    alpha = 0.001

    Q_func_basis(theta, s, a) = Q_func(basis, theta, s, a)
    gradQ_func_basis(theta, s, a) = gradQ_func(basis, theta, s, a)

    qlearning = GradientQLearning(A, gamma, Q_func_basis, gradQ_func_basis, theta, alpha)

    # Play games
    current_policy(s, usable_actions) = exploration_policy(qlearning, s, usable_actions)
    vals = py"play_game"(current_policy)

    print(vals)
end

main()