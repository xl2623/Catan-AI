using PyCall
using Flux
using BSON: @load
using LinearAlgebra

"""
    Catan Game Wrappers
"""

pushfirst!(pyimport("sys")."path", "")

play_game_with_policy = pyimport("AIGame_Wrapper")["play_game_with_policy"]

function play_game(policy)
    s, usable_a, a, r, sp, usable_ap, ap, rp, spp = play_game_with_policy(policy)

    # Convert to Julia style vectors
    s = convert_array_2_julia(s, Int64)
    usable_a = convert_array_2_julia(usable_a, Int64)
    sp = convert_array_2_julia(sp, Int64)
    usable_ap = convert_array_2_julia(usable_ap, Int64)
    spp = convert_array_2_julia(spp, Int64)

    return s, usable_a, a, r, sp, usable_ap, ap, rp, spp
end

function convert_array_2_julia(arr, convert_to)
    return map(x -> convert(convert_to, x), arr)
end

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

function epsilonGreedyExploration(model, epsilon, s, usable_actions)
    # TODO: There may be an issue with the indeces being 0-based in Python
    s = convert_array_2_julia(s, Int64)

    if rand() < epsilon
        return rand(usable_actions)
    end

    Q(s,a) = lookahead(model, s, a)
    return argmax(a->Q(s,a), usable_actions)
end

"""
    Main function
"""

function improve_theta(model, policy_fcn, epsilon, k, print_freq)
    wins = 0
    total_reward = 0

    for i in 1:k
        # Policy
        current_policy(s, usable_actions) = policy_fcn(model, epsilon, s, usable_actions)
        
        # Play one game
        s, usable_a, a, r, sp, usable_ap, ap, rp, spp = play_game(current_policy)

        # Update theta
        model = update!(model, s, a, r, sp, usable_a, false)
        model = update!(model, sp, ap, rp, spp, usable_ap, true)

        # Record results
        wins = rp == 0 ? wins + 1 : wins 
        total_reward += rp

        if i % print_freq == 0
            avg_reward = total_reward / print_freq

            println("##################################################")
            println("Number of iterations: $i")
            println("Number of wins: $wins")
            println("Total reward: $total_reward")
            println("Average reward: $avg_reward")
            println("##################################################")

            wins = 0
            total_reward = 0
        end
    end
end

function main()
    # Basis function
    # TODO: Update to use the pre-computed model
    basis = Chain(
        Dense(56, 32, Flux.relu), # TODO: Update once we add roads (from 56 to 57)
        Dense(32, 16, Flux.relu),
        Dense(16, 1),
    )
    # OR
    # @load "test.bson" basis
    # println(basis)
    # exit()

    # Gradient Q-Learning
    # TODO: uopdate action space
    # A = [[i, j] for j = 1:72 for i=1:54]
    A = 1:54
    gamma = 0.99
    theta = Flux.params(basis)
    alpha = 0.001

    Q_func_basis(theta, s, a) = Q_func(basis, theta, s, a)
    gradQ_func_basis(theta, s, a) = gradQ_func(basis, theta, s, a)

    qlearning = GradientQLearning(A, gamma, Q_func_basis, gradQ_func_basis, theta, alpha)

    # Play games
    k = 10000
    print_frequency = 1000
    epsilon = 0.1
    improve_theta(qlearning, epsilonGreedyExploration, epsilon, k, print_frequency)
end

main()