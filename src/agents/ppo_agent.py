import torch

from .a2c_agent import A2CAgent

class PPOAgent(A2CAgent):
    def __init__(self, cfg):
        super(PPOAgent, self).__init__(cfg)

    def update(self):
        cfg = self.cfg

        for epoch in range(cfg.mini_epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch, adv_batch = batch_data

                vs, pis = self.network(obs_batch)

                value_loss = (vs - gae_batch - value_batch).pow(2)
                vs_clipped = value_batch + (vs - value_batch).clamp(-cfg.clip_param, cfg.clip_param)
                vs_loss_clipped = (vs_clipped - gae_batch - value_batch).pow(2)
                value_loss = 0.5 * torch.max(value_loss, vs_loss_clipped).mean()

                dist, action_log_probs, entropy = self.pdist(pis, action_batch)

                ratio = torch.exp(action_log_probs - action_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv_batch
                policy_loss = torch.min(surr1, surr2).mean().neg()

                loss = value_loss * cfg.value_loss_coef + policy_loss - entropy * cfg.entropy_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                self.logger.store(
                    Loss=loss,
                    VLoss=value_loss,
                    PLoss=policy_loss,
                    Entropy=entropy
                )








