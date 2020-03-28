import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from copy import deepcopy
import numpy as np
from .ppo_agent import PPOAgent

class TRPOAgent(PPOAgent):
    def __init__(self, cfg):
        super(TRPOAgent, self).__init__(cfg)




    def surrogate_loss(self, params, obs_batch, action_batch, action_log_probs_old, adv_batch):
        model_new = deepcopy(self.network)
        vector_to_parameters(params, model_new.get_policy_params())
        _, logits = model_new(obs_batch)
        dist = Categorical(logits=logits)
        action_log_probs_new = dist.log_prob(action_batch.suqeeze(-1)).unsqueeze(-1)
        surr_loss = torch.exp(action_log_probs_new - action_log_probs_old) * adv_batch
        return surr_loss.neg().mean()

    def line_search(self, params, fullstep, expected_improve_rate, obs_batch, action_batch, action_log_probs, adv_batch):
        accept_ratio = 0.1
        max_backtracks = 10

        fval = self.surrogate_loss(params, obs_batch, action_batch, action_log_probs, adv_batch)

        for (n, step_frac) in enumerate(0.5 ** np.arange(max_backtracks)):
            print(f"Search number {n}")
            params_new = params.clone() + step_frac * fullstep
            new_fval = self.surrogate_loss(params_new, obs_batch, action_batch, action_log_probs, adv_batch)
            actual_improve = fval - new_fval
            expected_improve_rate = expected_improve_rate * step_frac
            ratio = actual_improve / expected_improve_rate
            if ratio > accept_ratio and actual_improve > 0:
                return params_new
        return params


    def hessian_vector_product(self, vector, obs_batch):
        self.optimizer.zero_grad()
        _, logits = self.network(obs_batch)
        kld_loss = F.kl_div(F.log_softmax(logits), F.softmax(logits).detach())

        kld_grad = torch.autograd.grad(kld_loss, self.network.get_policy_params(), create_graph=True)
        kdl_grad_vector = torch.cat([g.view(-1) for g in kld_grad])
        grad_vector_product = torch.sum(kdl_grad_vector * vector)

        grad_grad = torch.autograd.grad(grad_vector_product, self.network.get_policy_params())
        fisher_vector_product = torch.cat([g.contiguous().view(-1) for g in grad_grad]).detach()
        return fisher_vector_product + (self.cfg.cg_damping * vector.detach())

    def conjugate_gradient(self, grads, obs_batch):
        cfg = self.cfg

        p = grads.clone().double()
        r = grads.clone().double()
        x = torch.zeros_like(r).double()
        rdotr = r.dot(r)
        for _ in range(cfg.cg_iters):
            z = self.hessian_vector_product(p, obs_batch).squeeze()
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z

            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < cfg.residual_tol:
                break
        return x


    def update(self):

        cfg = self.cfg

        for epoch in range(cfg.epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, adv_batch = batch_data

                vs, pis = self.network(obs_batch)
                dist = Categorical(logits=pis)
                log_probs = dist.log_prob(action_batch.view(-1)).unsqueeze(-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - action_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv_batch
                policy_loss = torch.min(surr1, surr2).mean().neg() + entropy.neg() * cfg.entropy_coef

                self.optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                grads = parameters_to_vector([v.grad for v in self.network.get_policy_params()]).squeeze()

                step_direction = self.conjugate_gradient(grads.neg(), obs_batch)
                shs = 0.5 * step_direction.dot(self.hessian_vector_product(step_direction, obs_batch))
                lm = torch.sqrt(shs / cfg.max_kl)
                fullstep = step_direction / lm
                g_dot_step_direction = grads.double().neg().dot(step_direction).detach()
                theta = self.line_search(parameters_to_vector(self.network.parameters()))

                old_model = deepcopy(self.network)
                if torch.isnan(theta):
                    print("NAN detected")
                else:
                    vector_to_parameters(theta, self.network.get_policy_params())

                _, logits = self.network(obs_batch)
                kld = F.kl_div(F.log_softmax(logits), F.softmax(pis.detach()))
                print(f"KL: {kld.item()}")
        #         theta = self.linesearch()
        #
        #         value_loss = (vs - return_batch).pow(2)
        #
        #         vs_clipped = value_batch + (vs - value_batch).clamp(-cfg.clip_param, cfg.clip_param)
        #         vs_loss_clipped = (vs_clipped - return_batch).pow(2)
        #
        #         value_loss = 0.5 * torch.max(value_loss, vs_loss_clipped).mean()
        #
        #         loss = value_loss * cfg.value_loss_coef + policy_loss - entropy * cfg.entropy_coef
        #
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
        #         self.optimizer.step()
        #
        #         self.logger.store(Loss=loss.item())
        #         self.logger.store(VLoss=value_loss.item())
        #         self.logger.store(PLoss=policy_loss.item())
        #         self.logger.store(Entropy=entropy)
        #         self.logger.store(Loss=loss)
        #
        # self.rollouts.obs[0].copy_(self.rollouts.obs[-1])
        # self.rollouts.masks[0].copy_(self.rollouts.masks[-1])
        #
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
