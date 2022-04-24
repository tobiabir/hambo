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
        state_next, reward, done, _ = env.step(action)
        
        mask = 0. if idx_step + 1 == env.max_steps_episode else float(done)
        dataset.append(state, action, reward, state_next, mask)

        state = state_next

        reward_episode += reward
        idx_step += 1

    return dataset, state_initial, reward_episode
