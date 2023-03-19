using PyCall
using Flux
using BSON: @load, @save
using LinearAlgebra
using ProgressBars
using DataFrames
using CSV

"""
    Catan Game Wrappers
"""

pushfirst!(pyimport("sys")."path", "")

play_game_with_policy = pyimport("AIGame_Wrapper")["play_game_with_policy"]

function play_game(policy)
    return play_game(policy, "heuristic")    
end

function play_game(policy, other_player_type, constant_board=false, board=nothing)
    if constant_board
        s, usable_a, a, r, sp, usable_ap, ap, rp, spp, board = play_game_with_policy(policy, other_player_type, constant_board, board)
    else
        s, usable_a, a, r, sp, usable_ap, ap, rp, spp = play_game_with_policy(policy, other_player_type)
    end

    # Convert to Julia style vectors
    s = convert_array_2_julia(s, Int64)
    usable_a = convert_array_2_julia(usable_a, Int64)
    sp = convert_array_2_julia(sp, Int64)
    usable_ap = convert_array_2_julia(usable_ap, Int64)
    spp = convert_array_2_julia(spp, Int64)

    if constant_board
        return s, usable_a, a, r, sp, usable_ap, ap, rp, spp, board 
    else
        return s, usable_a, a, r, sp, usable_ap, ap, rp, spp
    end
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
        expected_utility(theta, sp, ap) = vec(ap) in eachrow(usable_actions) ? Q(theta,sp,ap) : -Inf
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
    params(basis) = theta
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
    s = convert_array_2_julia(s, Int64)

    if rand() < epsilon
        return usable_actions[rand(1:size(usable_actions)[1]), :]
    end

    Q(s,a) = lookahead(model, s, a)
    max_q = -Inf 
    max_action = []

    for action in eachrow(usable_actions)
        q = Q(s,action)
        if q > max_q
            max_q = q 
            max_action = action
        end
    end

    return max_action
end

"""
    Main function
"""

function improve_theta(model, policy_fcn, epsilon, k, print_freq, switch_player_type=Inf, train=true, constant_board=false,  decay_rate=1, decay_freq=Inf)
    wins = 0
    total_reward = 0
    initial_theta = deepcopy(model.theta)
    board = nothing

    delta_theta = ones(6,1)
    tot_theta_change = 0

    results = DataFrame([[],[],[],[],[]], ["iteration", "tot_theta_change", "win", "reward", "epsilon"])
    i = 1

    while i < k
        # Policy
        current_policy(s, usable_actions) = policy_fcn(model, epsilon, s, usable_actions)
        
        # Play one game
        if i < switch_player_type
            other_player_type = "random"
        else
            other_player_type = "heuristic"
        end
        
        if constant_board
            s, usable_a, a, r, sp, usable_ap, ap, rp, spp, board = play_game(current_policy, other_player_type, constant_board, board)
        else
            s, usable_a, a, r, sp, usable_ap, ap, rp, spp = play_game(current_policy, other_player_type)
        end

        # Update theta
        if rp > -11
            if train
                model = update!(model, s, a, r, sp, usable_a, false)
                model = update!(model, sp, ap, rp, spp, usable_ap, true)
            end

            # Record results
            wins = rp >= 0 ? wins + 1 : wins 
            total_reward += rp

            for i in 1:6
                delta_theta[i] = norm(model.theta[i] - initial_theta[i])
            end
            tot_theta_change = norm(delta_theta)

            push!(results, [i, tot_theta_change, rp>=0 ? 1 : 0, rp, epsilon])
            i += 1
        else
            continue
        end

        # Update epsilon
        if i % decay_freq == 0 && i != 0
            epsilon = epsilon * decay_rate
        end
        
        # Print results once in a while
        if i % print_freq == 0 || i == print_freq/10 || i == print_freq/2
            avg_reward = total_reward / print_freq

            println("##################################################")
            println("Number of iterations: $i")
            println("Number of wins: $wins")
            println("Total reward: $total_reward")
            println("Average reward: $avg_reward")
            println("Epsilon: $epsilon")
            # println("Change in theta: $delta_theta")
            println("Total change in theta: $tot_theta_change")
            println("##################################################")
            
            if i % print_freq == 0
                wins = 0
                total_reward = 0
            end
        end
    end

    return results
end

function main()
    # Basis function
    # TODO: Update to use the pre-computed model
    basis = Chain(
        Dense(57, 32, Flux.relu), # TODO: Update once we add roads (from 56 to 57)
        Dense(32, 16, Flux.relu),
        Dense(16, 1),
    )
    # OR
    # @load "alpha_1e-2_gamma_99e-2_epoch_10.bson" basis

    # Gradient Q-Learning
    A = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 3], [1, 4], [2, 3], [2, 5], [2, 6], [3, 5], [3, 7], [3, 8], [4, 7], [4, 9], [4, 10], [5, 1], [5, 9], [5, 11], [6, 4], [6, 12], [6, 13], [7, 12], [7, 14], [7, 15], [8, 14], [8, 16], [8, 17], [9, 6], [9, 16], [9, 18], [10, 19], [10, 20], [10, 21], [11, 13], [11, 19], [11, 22], [12, 2], [12, 20], [12, 23], [13, 23], [13, 24], [13, 25], [14, 11], [14, 26], [14, 27], [15, 24], [15, 26], [15, 28], [16, 10], [16, 29], [16, 30], [17, 29], [17, 31], [17, 32], [18, 27], [18, 31], [18, 33], [19, 8], [19, 34], [19, 35], [20, 34], [20, 36], [20, 37], [21, 30], [21, 36], [21, 38], [22, 18], [22, 39], [22, 40], [23, 35], [23, 39], [23, 41], [24, 15], [24, 42], [24, 43], [25, 42], [25, 44], [26, 44], [26, 45], [27, 17], [27, 45], [27, 46], [28, 22], [28, 47], [28, 48], [29, 43], [29, 47], [30, 49], [30, 50], [31, 48], [31, 49], [32, 21], [32, 50], [32, 51], [33, 51], [33, 52], [34, 25], [34, 52], [34, 53], [35, 53], [35, 54], [36, 28], [36, 55], [36, 56], [37, 54], [37, 55], [38, 33], [38, 57], [38, 58], [39, 56], [39, 57], [40, 32], [40, 59], [40, 60], [41, 59], [41, 61], [42, 58], [42, 61], [43, 38], [43, 62], [43, 63], [44, 60], [44, 62], [45, 37], [45, 64], [45, 65], [46, 64], [46, 66], [47, 63], [47, 66], [48, 41], [48, 67], [48, 68], [49, 65], [49, 67], [50, 40], [50, 69], [50, 70], [51, 69], [51, 71], [52, 68], [52, 71], [53, 46], [53, 70]]
    # A = 1:54
    gamma = 0.99
    theta = Flux.params(basis)
    alpha = 0.01

    Q_func_basis(theta, s, a) = Q_func(basis, theta, s, a)
    gradQ_func_basis(theta, s, a) = gradQ_func(basis, theta, s, a)

    qlearning = GradientQLearning(A, gamma, Q_func_basis, gradQ_func_basis, theta, alpha)

    # Play games
    k = 50000
    print_frequency = 1000
    epsilon = 1.0
    decay_rate = 0.9
    decay_freq = 500
    switch_player_type = Inf        # Iteration at which we switch the opponents from random to heurisitc
    train = true
    constant_board = true
    results = improve_theta(qlearning, epsilonGreedyExploration, epsilon, k, print_frequency, switch_player_type, train, constant_board, decay_rate, decay_freq)

    filename = "./data/results_alpha_1e-2_epsilon_1_decay_9e-1_freq_1000"
    CSV.write(filename*".csv", results)
    @save filename*".bson" basis
end

main()