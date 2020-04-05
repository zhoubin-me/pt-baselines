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


    @staticmethod
    def conjugate_gradients(fvp_func, b, iters):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        new_rnorm = torch.dot(r, r)
        for _ in range(iters):
            rnorm = new_rnorm
            fvp = fvp_func(p)
            alpha = rnorm / torch.dot(p, fvp)
            x += alpha * p
            r -= alpha * fvp
            new_rnorm = torch.dot(r, r)
            ratio = new_rnorm / rnorm
            p = r + ratio * p
        return x

    @staticmethod
    def line_search(f, x, expected_improve_rate, trials, accept_ratio):
        for i in range(trials):
            scaling = 2 ** (-i)
            scaled = x * scaling
            improve = f(scaled)
            expected_improve_rate = expected_improve_rate * scaling
            if improve / expected_improve_rate > accept_ratio and improve > 0:
                print("We good! %f" % (scaling,))
                return scaled

        print("We are bad!")
        return 0

    @staticmethod
    def KL(pdist, qdist=None):
        if isinstance(pdist, Normal):
            p_mean, p_std = pdist.loc, pdist.scale
            d = p_mean.shape[0]
            if qdist is None:
                q_mean, q_std = p_mean.detach(), p_std.detach()
            else:
                q_mean, q_std = qdist.loc.detach(), qdist.scale.detach()

            detp = p_std.log().sum(-1, keepdim=True).exp()
            detq = q_std.log().sum(-1, keepdim=True).exp()
            diff = q_mean - p_mean

            log_quot_frac = detp.log() - detq.log()
            tr = (p_std / q_std).sum(-1, keepdim=True)
            quadratic = ((diff / q_std) * diff).sum(-1, keepdim=True)

            kl = 0.5 * (log_quot_frac - d + tr + quadratic)
            kl = kl.mean()

        elif isinstance(pdist, Categorical):
            kl = F.kl_div(pdist.logits.log_softmax(-1), pdist.probs.detach(), reduction='batchmean')
        else:
            raise NotImplementedError("KL not implemented")
        return kl

    def update(self):
        cfg = self.cfg
        # Value Step
        for epoch in range(cfg.mini_epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch, adv_batch = batch_data

                vs = self.network.v(obs_batch)
                value_loss = (vs - value_batch - gae_batch).pow(2).mean()

                self.optimizer.zero_grad()
                value_loss.backward()
                self.optimizer.step()

        # Policy Update
        sampler = self.sample(cfg.num_processes * cfg.mini_steps)
        for batch_data in sampler:
            obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch, adv_batch = batch_data

            vs, pis = self.network(obs_batch)
            if isinstance(self.envs.action_space, Discrete):
                pdist = Categorical(logits=pis)
                log_prob = pdist.log_prob(action_batch.view(-1)).unsqueeze(-1)
                entropy = pdist.entropy().mean()
                kl = self.KL(pdist)
            elif isinstance(self.envs.action_space, Box):
                pdist = Normal(pis, self.network.p_log_std.expand_as(pis).exp())
                log_prob = pdist.log_prob(action_batch).sum(-1, keepdim=True)
                entropy = pdist.entropy().sum(-1).mean()
                kl = self.KL(pdist)
            else:
                raise NotImplementedError('No such action space')

            policy_loss_neg = (adv_batch * (log_prob - action_log_prob_batch).exp()).mean()
            grads = torch.autograd.grad(policy_loss_neg, self.network.get_policy_params(), retain_graph=True)
            grads = parameters_to_vector(grads)

            g = torch.autograd.grad(kl, self.network.get_policy_params(), create_graph=True)
            g = parameters_to_vector(g)

            def fisher_product(x, cg_damping=cfg.cg_damping):
                z = g @ x
                hv = torch.autograd.grad(z, self.network.get_policy_params(), retain_graph=True)
                return torch.cat([v.contiguous().view(-1) for v in hv]).detach() * x * cg_damping

            step = self.conjugate_gradients(fisher_product, grads, cfg.cg_iters)

            max_step_coeff = (2 * cfg.max_kl / (step @ fisher_product(step))) ** (0.5)
            max_trpo_step = max_step_coeff * step

            with torch.no_grad():
                value_loss = (vs - value_batch - gae_batch).pow(2).mean()
                params_old = parameters_to_vector(self.network.get_policy_params())
                expected_improve = grads & max_trpo_step

                def backtrac_fn(s):
                    params_new = params_old + s
                    vector_to_parameters(params_new, self.network.get_policy_params())
                    pis_new = self.network.p(obs_batch)

                    pdist_new = Normal(pis_new, self.network.p_log_std.expand_as(pis).exp())
                    log_prob_new = pdist_new.log_prob(action_batch).sum(-1, keepdim=True)
                    policy_loss_neg_new = (adv_batch * (log_prob_new - action_log_prob_batch).exp()).mean()
                    kl_new = self.KL(pdist, pdist_new)

                    if policy_loss_neg_new <= policy_loss_neg or kl_new > cfg.max_kl:
                        return -float('inf')
                    return policy_loss_neg_new - policy_loss_neg

                final_step = self.line_search(backtrac_fn, max_trpo_step, expected_improve, cfg.max_backtracks, cfg.accept_ratio)

                new_params = params_old + final_step
                vector_to_parameters(new_params, self.network.get_policy_params())

            kwargs = {
                'Loss': 0,
                'VLoss': value_loss.item(),
                'PLoss': policy_loss_neg.item(),
                'Entropy': entropy.item()
            }

            self.logger.store(**kwargs)

                    # fval = self.policy_loss(states, actions, advs, old_log_prbs).detach()
                    # for (_n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
                    #     xnew = x + stepfrac * fullstep
                    #     vector_to_parameters(xnew, self.network.get_policy_params())
                    #     newfval = self.policy_loss(states, actions, advs, old_log_prbs).detach()
                    #     actual_improve = fval - newfval
                    #     expected_improve = expected_improve_rate * stepfrac
                    #     ratio = actual_improve / expected_improve
                    #
                    #     if ratio.item() > accept_ratio and actual_improve.item() > 0:
                    #         return True, xnew
                    #
                    # return False, x

        # entropy = pdist.entropy().mean()
        # policy_loss = self.policy_loss(obs_batch, action_batch, adv_batch, fixed_log_prob)
        # grads = torch.autograd.grad(policy_loss, self.network.get_policy_params())
        # loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        #
        # stepdir = self.conjugate_gradients(-loss_grad, obs_batch)
        # shs = 0.5 * (stepdir * self.Fvp(stepdir, obs_batch)).sum(0, keepdim=True)
        #
        # lm = torch.sqrt(shs / self.cfg.max_kl)
        # fullstep = stepdir / lm.detach()
        #
        # neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        # prev_params = parameters_to_vector(self.network.get_policy_params())
        #
        # with torch.no_grad():
        #     success, new_params = self.linesearch(obs_batch, action_batch, adv_batch, fixed_log_prob, prev_params, fullstep, neggdotstepdir / lm)
        #
        # vector_to_parameters(new_params, self.network.get_policy_params())
        #

