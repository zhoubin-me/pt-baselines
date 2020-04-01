import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np
from .ppo_agent import A2CAgent

class TRPOAgent(A2CAgent):
    def __init__(self, cfg):
        super(TRPOAgent, self).__init__(cfg)

    def vloss_closure(self, states, targets):
        values_ = self.network.v(states)
        v_loss = 0.5 * (values_ - targets).pow(2).mean()
        for param in self.network.v.parameters():
            v_loss += param.pow(2).sum() * self.cfg.l2_reg
        self.optimizer.zero_grad()
        v_loss.backward()
        return v_loss

    def get_ploss(self, states, actions, advs, old_log_probs, volatile=False):
        if volatile:
            with torch.no_grad():
                dist = self.network.pdist(states)
        else:
            dist = self.network.pdist(states)

        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        action_loss = -advs * torch.exp(log_prob - old_log_probs)
        return action_loss.mean()

    def get_kl(self, state):
        mean1 = self.network.p(state)
        log_std1 = self.network.p_log_std.expand_as(mean1)
        std1 = log_std1.exp()
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def Fvp(self, v, state):
        kl = self.get_kl(state)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, self.network.get_policy_params(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, self.network.get_policy_params())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * self.cfg.cg_damping

    def conjugate_gradients(self, b, state, residual_tol=1e-10):
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
            if rdotr < residual_tol:
                break
        return x


    def linesearch(self, states, actions, advs, old_log_prbs, x,
                   fullstep,
                   expected_improve_rate,
                   max_backtracks=10,
                   accept_ratio=.1):
        fval = self.get_ploss(states, actions, advs, old_log_prbs, True).detach()
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            vector_to_parameters(xnew, self.network.get_policy_params())
            newfval = self.get_ploss(states, actions, advs, old_log_prbs, True).detach()
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
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch = batch_data
                adv_batch = (gae_batch - gae_batch.mean()) / (gae_batch.std() + 1e-5)

                for d in batch_data:
                    if torch.any(torch.isnan(d)):
                        import pdb
                        pdb.set_trace()

                v_params_old = parameters_to_vector(self.network.get_value_params())
                self.optimizer.step(lambda: self.vloss_closure(obs_batch, gae_batch + value_batch))
                value_loss = self.vloss_closure(obs_batch, gae_batch + value_batch)
                if value_loss.item() > 100:
                    vector_to_parameters(v_params_old, self.network.get_value_params())


                pdist = self.network.pdist(obs_batch)
                fixed_log_prob = pdist.log_prob(action_batch).sum(-1, keepdim=True).detach()
                policy_loss = self.get_ploss(obs_batch, action_batch, adv_batch, fixed_log_prob)

                grads = torch.autograd.grad(policy_loss, self.network.get_policy_params())
                loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

                stepdir = self.conjugate_gradients(-loss_grad, obs_batch)
                shs = 0.5 * (stepdir * self.Fvp(stepdir, obs_batch)).sum(0, keepdim=True)

                lm = torch.sqrt(shs / self.cfg.max_kl)
                fullstep = stepdir / lm.detach()

                neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
                prev_params = parameters_to_vector(self.network.get_policy_params())
                success, new_params = self.linesearch(obs_batch, action_batch, adv_batch, fixed_log_prob, prev_params, fullstep, neggdotstepdir / lm)

                vector_to_parameters(new_params, self.network.get_policy_params())

                kwargs = {
                    'Loss': 0,
                    'VLoss': value_loss.item(),
                    'PLoss': policy_loss.item(),
                    'Entropy': 0,
                }
                self.logger.store(**kwargs)
