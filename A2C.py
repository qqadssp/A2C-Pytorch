import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import time
import os
from runner import Runner
from util import logger

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def constfn(val):
    def f(_):
        return val
    return f

def train(agent, env, N_steps, N_updates, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, epsilon=1e-5, alpha=0.95,
            log_interval=10, batch_size=4, N_train_sample_epochs=4, cliprange=0.2,
            save_interval=0):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)

    runner = Runner(env, agent, nsteps=N_steps, gamma=gamma, lam=lam)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    for update in range(1, N_updates+1):

        obs, returns, dones, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)

        frac = 1.0 - (update - 1.0) / N_updates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        optimazer = optim.RMSprop(agent.parameters(), lr=lrnow, weight_decay=alpha, eps=epsilon)

        mblossnames = ['policy_loss', 'value_loss', 'entropy']
        mblossvals = []

        agent.train()

        obs_ = torch.tensor(obs, requires_grad=True).float()
        returns_ = torch.tensor(returns).float()
        actions_ = torch.tensor(actions).float()
        values_ = torch.tensor(values).float()
        neglogpacs_ = torch.tensor(neglogpacs).float()
        advs_ = returns_ - values_

        optimazer.zero_grad()
        neglogp, entropy, vpred = agent.statistics(obs_, actions_)
        entropy = torch.mean(entropy)
        vf_loss = torch.mean(0.5 * (vpred - returns_) ** 2)
        pg_loss = torch.mean(advs_ * neglogp)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        loss.backward()
        optimazer.step()


        mblossvals.append([pg_loss.item(), vf_loss.item(), entropy.item()])

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        N_sample_steps = obs.shape[0]
        fps = int(update * N_sample_steps / (tnow - tfirststart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*N_sample_steps)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, mblossnames):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = os.path.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            torch.save(agent.state_dict(), savepath)
    env.close()
    return agent

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

