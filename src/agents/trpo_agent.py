import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from gym.spaces import Box, Discrete
import numpy as np
from .ppo_agent import A2CAgent

class TRPOAgent(A2CAgent):
    def __init__(self, cfg):
        super(TRPOAgent, self).__init__(cfg)

    def policy_loss(self, states, actions, advs, old_log_probs):
        pis = self.network.p(states)
        if isinstance(self.envs.action_space, Discrete):
            pdist = Categorical(logits=pis)
            log_prob = pdist.log_prob(actions.view(-1)).unsqueeze(-1)
            entropy = pdist.entropy().mean()
        elif isinstance(self.envs.action_space, Box):
            pdist = Normal(pis, self.network.p_log_std.expand_as(pis).exp())
            log_prob = pdist.log_prob(actions).sum(-1, keepdim=True)
            entropy = pdist.entropy().sum(-1).mean()
        else:
            raise NotImplementedError('No such action space')

        action_loss = advs.neg() * torch.exp(log_prob - old_log_probs) + self.cfg.entropy_coef * entropy.neg()
        return action_loss.mean()

    def get_kl(self, state):

        if isinstance(self.envs.action_space, Discrete):
            pis = self.network.p(state)
            kl = F.kl_div(pis.log_softmax(dim=-1), (pis + 1e-8).softmax(dim=-1).detach(), reduction='batchmean')
        elif isinstance(self.envs.action_space, Box):
            mean1 = self.network.p(state)
            log_std1 = self.network.p_log_std.expand_as(mean1)
            std1 = log_std1.exp()
            mean0 = mean1.detach()
            log_std0 = log_std1.detach()
            std0 = std1.detach()
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            kl = kl.sum(1, keepdim=True)
        else:
            raise NotImplementedError('No such action space')
        return kl

    def Fvp(self, v, state):
        kl = self.get_kl(state)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, self.network.get_policy_params(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, self.network.get_policy_params())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * self.cfg.cg_damping

    def conjugate_gradients(self, b, state):
        x = torch.zeros_like(b)
        r = b - self.Fvp(x, state)
        p = r
        rdotr = torch.dot(r, r)

        for i in range(self.cfg.cg_iters):
            _Avp = self.Fvp(p, state)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < self.cfg.residual_tol:
                break
        return x


    def linesearch(self, states, actions, advs, old_log_prbs, x,
                   fullstep,
                   expected_improve_rate,
                   max_backtracks=10,
                   accept_ratio=.1):
        fval = self.policy_loss(states, actions, advs, old_log_prbs).detach()
        for (_n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            vector_to_parameters(xnew, self.network.get_policy_params())
            newfval = self.policy_loss(states, actions, advs, old_log_prbs).detach()
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return True, xnew

        return False, x


    def update(self):
        cfg = self.cfg
        for epoch in range(cfg.mini_epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch, adv_batch = batch_data

                vs, pis = self.network(obs_batch)
                value_loss = (vs - return_batch - gae_batch).pow(2).mean() * 0.5

                self.optimizer.zero_grad()
                value_loss.backward()
                self.optimizer.step()

                if isinstance(self.envs.action_space, Discrete):
                    pdist = Categorical(logits=pis)
                    fixed_log_prob = pdist.log_prob(action_batch.view(-1)).unsqueeze(-1).detach()
                    entropy = pdist.entropy().mean()
                elif isinstance(self.envs.action_space, Box):
                    pdist = Normal(pis, self.network.p_log_std.expand_as(pis).exp())
                    fixed_log_prob = pdist.log_prob(action_batch).sum(-1, keepdim=True).detach()
                    entropy = pdist.entropy().sum(-1).mean()
                else:
                    raise NotImplementedError('No such action space')

                entropy = pdist.entropy().mean()
                policy_loss = self.policy_loss(obs_batch, action_batch, adv_batch, fixed_log_prob)
                grads = torch.autograd.grad(policy_loss, self.network.get_policy_params())
                loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

                stepdir = self.conjugate_gradients(-loss_grad, obs_batch)
                shs = 0.5 * (stepdir * self.Fvp(stepdir, obs_batch)).sum(0, keepdim=True)

                lm = torch.sqrt(shs / self.cfg.max_kl)
                fullstep = stepdir / lm.detach()

                neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
                prev_params = parameters_to_vector(self.network.get_policy_params())

                with torch.no_grad():
                    success, new_params = self.linesearch(obs_batch, action_batch, adv_batch, fixed_log_prob, prev_params, fullstep, neggdotstepdir / lm)

                vector_to_parameters(new_params, self.network.get_policy_params())

                kwargs = {
                    'Loss': 0,
                    'VLoss': value_loss.item(),
                    'PLoss': policy_loss.item(),
                    'Entropy': entropy.item()
                }
                self.logger.store(**kwargs)
