import numpy as np

class Runner(object):

    def __init__(self, env, agent, nsteps, gamma, lam):
        self.env = env
        self.agent = agent
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = agent.initial_state
        self.dones = [False for _ in range(nenv)]
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        self.agent.eval()
        for idx in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.agent.step(self.obs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        mb_dones.pop(0)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        _, last_values, _, _ = self.agent.step(self.obs)
        last_values = last_values * (1.0 - self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)

        r = last_values
        for t in reversed(range(self.nsteps)):
            mb_returns[t] = r = mb_rewards[t] + self.gamma * r * (1.0 - mb_dones[t])

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


