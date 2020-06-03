import torch
import torch.nn.functional as F

from .a2c_agent import A2CAgent

class PPOAgent(A2CAgent):
    def __init__(self, cfg):
        super(PPOAgent, self).__init__(cfg)

    def update(self):
        cfg = self.cfg

        # Value Step
        for epoch in range(cfg.mini_epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch, adv_batch = batch_data

                sel = mask_batch.bool()
                vs = self.network.v(obs_batch)
                vs_target = value_batch + gae_batch
                value_loss = F.mse_loss(vs[sel], vs_target[sel])

                self.optimizer_v.zero_grad()
                value_loss.backward()
                self.optimizer_v.step()


        # Policy Step
        for epoch in range(cfg.mini_epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch, adv_batch = batch_data

                pis = self.network.p(obs_batch)

                dist, action_log_probs, entropy = self.pdist(pis, action_batch)

                ratio = torch.exp(action_log_probs - action_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv_batch
                policy_loss = torch.min(surr1, surr2).mean().neg()

                loss = policy_loss - entropy * cfg.entropy_coef

                self.optimizer_p.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer_p.step()

                self.logger.store(
                    Loss=loss,
                    PLoss=policy_loss,
                    Entropy=entropy
                )








