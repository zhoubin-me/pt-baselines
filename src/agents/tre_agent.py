import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from .ddpg_agent import DDPGAgent

class TD3Agent(DDPGAgent):
    def __init__(self, cfg):
        super(TD3Agent, self).__init__(cfg)


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
                return scaled
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

    def update(self, *args):
        states, actions, rewards, next_states, terminals = args
        cfg = self.cfg
        with torch.no_grad():
            next_actions, _, _, _ = self.target_network.act(next_states)
            target_q1, target_q2 = self.target_network.action_value(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - terminals) * cfg.gamma * target_q.detach()

        current_q1, current_q2 = self.network.action_value(states, actions)
        value_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        self.logger.store(VLoss=value_loss)
        sampled_action, pdist, _ = self.network.act(states)
        policy_adv = self.network.v(torch.cat([states, sampled_action], dim=1)).mean()
        kl = self.KL(pdist)

        grads = torch.autograd.grad(policy_adv, self.network.get_policy_params(), retain_graph=True)
        grads = parameters_to_vector(grads)

        g = torch.autograd.grad(kl, self.network.get_policy_params(), create_graph=True)
        g = parameters_to_vector(g)

        def fisher_product(x, cg_damping=cfg.cg_damping):
            z = g @ x
            hv = torch.autograd.grad(z, self.network.get_policy_params(), retain_graph=True)
            return torch.cat([v.contiguous().view(-1) for v in hv]).detach() + x * cg_damping

        step = self.conjugate_gradients(fisher_product, grads, 10)
        max_step_coef = (2 * cfg.max_kl / (step @ fisher_product(step))) ** (0.5)
        max_trpo_step = max_step_coef * step

        with torch.no_grad():
            params_old = parameters_to_vector(self.network.get_policy_params())
            expected_improve = grads @ max_trpo_step

            def backtrac_fn(s):
                params_new = params_old + s
                vector_to_parameters(params_new, self.network.get_policy_params())
                sampled_action_new, pdist_new, _ = self.network.act(states)
                policy_adv_new = self.network.v(torch.cat([states, sampled_action_new], dim=1)).mean()
                kl_new = self.KL(pdist, pdist_new)

                if policy_adv_new <= policy_adv or kl_new > cfg.max_kl:
                    return -float('inf')
                return policy_adv_new - policy_adv

            final_step = self.line_search(backtrac_fn, max_trpo_step, expected_improve, cfg.max_backtracks, cfg.accept_ratio)

            new_params = params_old + final_step
            vector_to_parameters(new_params, self.network.get_policy_params())





        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

        self.logger.store(PLoss=policy_adv.neg())
