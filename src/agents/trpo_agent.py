import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

import numpy as np
from .ppo_agent import PPOAgent


class TRPOAgent(PPOAgent):
    def __init__(self, cfg):
        super(TRPOAgent, self).__init__(cfg)

        self.value_optimizer = torch.optim.LBFGS(self.network.get_value_params(), max_iter=25)

    def update(self):
        cfg = self.cfg
        network = self.network
        logger = self.logger
        for epoch in range(cfg.epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, adv_batch = batch_data

                ## For Value Loss
                def vloss_closure():
                    self.value_optimizer.zero_grad()
                    v = network.v(obs_batch)
                    v_loss = (v - return_batch).view(-1, 1).pow(2).mean()
                    v_loss.backward()
                    for param in network.get_value_params():
                        v_loss += param.pow(2).sum() * cfg.l2_reg
                    return v_loss

                self.value_optimizer.step(vloss_closure)
                value_loss = vloss_closure()
                logger.store(VLoss=value_loss.item())

                ## For Policy Loss
                def ploss_closure():
                    dist = network.pdist(obs_batch)
                    action_log_probs = dist.log_prob(action_batch)
                    p_loss = (adv_batch * torch.exp(action_log_probs - action_log_prob_batch)).mean().neg()
                    return p_loss

                def kl_loss():
                    dist = network.pdist(obs_batch)
                    action_log_probs = dist.log_prob(action_batch)
                    kl_loss = F.kl_div(action_log_probs, action_log_prob_batch.exp())
                    return kl_loss

                def Avp(v):
                    kl = kl_loss()
                    grads = torch.autograd.grad(kl, network.get_policy_params(), create_graph=True)
                    grads = torch.cat([g.view(-1) for g in grads])
                    kl_v = (grads * v).sum()
                    grads_grads = torch.autograd.grad(kl_v, network.get_policy_params())
                    grads_grads = torch.cat([g.contiguous().view(-1) for g in grads_grads]).detach()
                    return grads_grads + cfg.cg_damping * v

                def conjugate_gradients(b):
                    x = torch.zeros_like(b)
                    r = b.clone()
                    p = b.clone()
                    rdotr = r.dot(r)
                    for i in range(cfg.cg_iters):
                        _avp = Avp(p)
                        alpha = rdotr / p.dot(_avp)
                        x += alpha * p
                        r -= alpha * _avp
                        new_rdotr = r.dot(r)
                        beta = new_rdotr / rdotr
                        p = r + beta * p
                        rdotr = new_rdotr
                        if rdotr < cfg.residual_tol:
                            break
                    return x


                def linesearch(x, fullstep, expected_improve_rate):

                    with torch.no_grad():
                        fval = ploss_closure()
                    for (n, step_frac) in enumerate(0.5 ** np.arange(cfg.max_backtracks)):
                        x_new = x + step_frac * fullstep
                        vector_to_parameters(x_new, network.get_policy_params())
                        with torch.no_grad():
                            new_fval = ploss_closure()

                            actual_improve = fval - new_fval
                            expected_improve = expected_improve_rate * step_frac
                            ratio = actual_improve / expected_improve

                            if ratio.item() > cfg.accept_ratio:
                                return True, x_new
                    return False, x


                policy_loss = ploss_closure()
                grads = torch.autograd.grad(policy_loss, network.get_policy_params(), allow_unused=True)
                grads = torch.cat([g.view(-1) for g in grads]).detach().neg()
                step_dir = conjugate_gradients(grads)
                shs = 0.5 * (step_dir * Avp(step_dir)).sum()
                lm = (shs / cfg.max_kl).sqrt()
                fullstep = step_dir / lm

                g_dot_dir = grads.dot(step_dir).neg()
                print(f"LM:{lm.item()}, Grad Norm:{grads.norm().item()}")

                prev_params = parameters_to_vector(network.get_policy_params())
                success, new_params = linesearch(prev_params, fullstep, g_dot_dir / lm)
                if not success:
                    print("Failed")
                vector_to_parameters(new_params, network.get_policy_params())
                kl_loss = kl_loss()

                logger.store(PLoss=policy_loss.item())
                logger.store(Entropy=kl_loss.item())
