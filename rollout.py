import data

def rollout_episode(env, agent):
    dim_action = env.action_space.shape[0]

    dataset = data.DatasetSARS()
    reward_episode = 0
    idx_step = 0

    state = env.reset()
    state_initial = state
    done = False
    while not done:
        action = agent.get_action(state)[:dim_action]
        state_next, reward, done, info = env.step(action)
        
        mask = float(done and not info["TimeLimit.truncated"])
        dataset.push(state, action, reward, state_next, mask)

        state = state_next

        reward_episode += reward
        idx_step += 1

    return dataset, state_initial, reward_episode

def rollout_steps(env, agent, dataset, dataset_states_initial, num_steps):
    dim_action = env.action_space.shape[0]
    state = env.state
    for idx_step in range(num_steps):
        action = agent.get_action(state)
        state_next, reward, done, info = env.step(action[:dim_action])
        mask = float(done and not info["TimeLimit.truncated"])
        dataset.push(state, action, reward, state_next, mask)
        state = state_next
        if done:
            state = env.reset()
        if dataset_states_initial is not None:
            dataset_states_initial.append(state)

