import torch
from torch.distributions import Categorical
from delegans.agents.a2c_agent import A2CAgent
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPOAgent(A2CAgent):
    def __init__(self, cfg):
        super(PPOAgent, self).__init__(cfg)

        self.optimizer = torch.optim.Adam(self.network.parameters(), cfg.lr, eps=cfg.eps)

        if cfg.use_lr_decay:
            num_updates = cfg.max_steps // (cfg.num_processes * cfg.nsteps)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: (num_updates - epoch) / num_updates)
        else:
            self.lr_scheduler = None

    def data_generator(self, advantages):
        cfg = self.cfg

        num_steps, num_processes = self.rollouts.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // cfg.num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = self.rollouts.obs[:-1].view(-1, *self.rollouts.obs.size()[2:])[indices]
            actions_batch = self.rollouts.actions.view(-1, self.rollouts.actions.size(-1))[indices]

            value_preds_batch = self.rollouts.values[:-1].view(-1, 1)[indices]
            return_batch = self.rollouts.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.rollouts.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.rollouts.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def update(self):
        cfg = self.cfg

        advantages = self.rollouts.returns[:-1] - self.rollouts.values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(cfg.epoches):
            data_generator = self.data_generator(advantages)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, pis = self.network(obs_batch)
                dist = Categorical(logits=pis)
                action_log_probs, dist_entropy = dist.log_prob(actions_batch.view(-1)).unsqueeze(-1), dist.entropy().mean()

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param,
                                    1.0 + cfg.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-cfg.clip_param, cfg.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                             value_losses_clipped).mean()


                loss = value_loss * cfg.value_loss_coef + action_loss - dist_entropy * cfg.entropy_coef
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                self.logger.store(VLoss=value_loss.item())
                self.logger.store(PLoss=action_loss.item())
                self.logger.store(Entropy=dist_entropy.item())
                self.logger.store(Loss=loss.item())

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        self.rollouts.obs[0].copy_(self.rollouts.obs[-1])
        self.rollouts.masks[0].copy_(self.rollouts.masks[-1])
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()




