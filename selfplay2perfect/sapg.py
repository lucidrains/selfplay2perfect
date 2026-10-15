from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from einops import rearrange, repeat

from torch_einops_utils import cast_tensor, masked_mean

from x_ppo import (
    calc_gae,
    clipped_value_loss_fn,
    dones_to_masks,
    ppo_actor_loss
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def stack_time(tensors):
    return torch.stack(tensors, dim = 1)

# distributed - the nodes of the population train one set of weights, each averaging into it

def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def groups_by_dtype(tensors):
    groups = {}

    for tensor in tensors:
        groups.setdefault(tensor.dtype, []).append(tensor)

    return list(groups.values())

def all_reduce_mean_(tensors, world = None):
    # in place mean over the process group, coalesced per dtype - one collective, not one per tensor
    # a no-op when the process group is not up, so a single node takes the same code path

    tensors = [tensor for tensor in tensors if exists(tensor)]

    if not tensors or not is_distributed():
        return

    world = default(world, torch.distributed.get_world_size())

    for group in groups_by_dtype(tensors):
        group = [tensor if tensor.is_contiguous() else tensor.contiguous() for tensor in group]

        flat = torch._utils._flatten_dense_tensors(group)
        torch.distributed.all_reduce(flat)
        flat /= world

        for tensor, mean in zip(group, torch._utils._unflatten_dense_tensors(flat, group)):
            tensor.copy_(mean)

# experience collected by rolling out the population

@dataclass
class Experience:
    # leading dimension is the environment, followed by time
    # the environments are split across blocks by block_ids

    obs: Tensor
    actions: Tensor
    log_probs: Tensor
    values: Tensor              # (env, time + 1) - the last value bootstraps the advantage
    rewards: Tensor             # (env, time)
    block_ids: Tensor           # (env,)
    priv_obs: Tensor | None = None
    aux: dict | None = None     # remaining observation keys - the scene, the past action - handed to the agent as is
    terminated: Tensor | None = None
    truncated: Tensor | None = None

    def to(self, device):
        return replace(
            self,
            obs = self.obs.to(device),
            actions = self.actions.to(device),
            log_probs = self.log_probs.to(device),
            values = self.values.to(device),
            rewards = self.rewards.to(device),
            block_ids = self.block_ids.to(device),
            priv_obs = self.priv_obs.to(device) if exists(self.priv_obs) else None,
            aux = {key: value.to(device) for key, value in self.aux.items()} if exists(self.aux) else None,
            terminated = self.terminated.to(device) if exists(self.terminated) else None,
            truncated = self.truncated.to(device) if exists(self.truncated) else None
        )

    def validate(self, num_blocks = None):
        # shapes that would otherwise ride into the loss as a silent miscalculation

        assert self.rewards.ndim == 2, f'rewards must be (env, time), got {tuple(self.rewards.shape)}'

        num_envs, time = self.rewards.shape

        for name in ('log_probs', 'terminated', 'truncated'):
            tensor = getattr(self, name)

            if exists(tensor):
                assert tensor.shape == (num_envs, time), f'{name} must be {(num_envs, time)}, got {tuple(tensor.shape)}'

        for name in ('obs', 'actions', 'priv_obs'):
            tensor = getattr(self, name)

            if exists(tensor):
                assert tensor.shape[:2] == (num_envs, time), f'{name} must lead with {(num_envs, time)}, got {tuple(tensor.shape[:2])}'

        for key, value in default(self.aux, dict()).items():
            assert value.shape[:2] == (num_envs, time), f'{key} in aux must lead with {(num_envs, time)}, got {tuple(value.shape[:2])}'

        assert self.values.shape == (num_envs, time + 1), f'values must be (env, time + 1) - the last value bootstraps - got {tuple(self.values.shape)}'
        assert self.block_ids.shape == (num_envs,), f'block_ids must be (env,), got {tuple(self.block_ids.shape)}'

        if exists(num_blocks):
            assert 0 <= int(self.block_ids.min()) and int(self.block_ids.max()) < num_blocks, f'block_ids must lie in [0, {num_blocks})'

# sapg losses - the per block surrogates and the leader's aggregation, both composed from the x-ppo functions

class SAPGLoss(NamedTuple):
    loss: Tensor
    policy_loss: Tensor
    off_policy_loss: Tensor
    value_loss: Tensor
    entropy_per_block: Tensor

def per_block_ppo_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    block_ids,
    num_blocks,
    eps_clip = 0.1,
    normalize_advantages = True
):
    # one surrogate per block of the population - with a single block, this is plain ppo

    losses = []

    for block_index in range(num_blocks):
        mask = rearrange(block_ids == block_index, 'e -> e 1').expand_as(action_log_probs)

        losses.append(ppo_actor_loss(
            action_log_probs,
            old_action_log_probs,
            advantages,
            eps_clip = eps_clip,
            mask = mask,
            normalize_advantages = normalize_advantages
        ))

    return torch.stack(losses)

def block_entropy(entropy, block_ids, num_blocks):
    return torch.stack([masked_mean(entropy, mask = block_ids == block_index) for block_index in range(num_blocks)])

def leader_off_policy_loss(
    leader_log_prob,
    follower_log_prob,
    follower_advantages,
    num_samples,
    eps_clip = 0.1,
    normalize_advantages = True
):
    # truncated importance sampling with mu = pi_leader / pi_follower, subsampled to the leader's on policy count
    # the old log probs are shifted by log mu, so that the x-ppo ratio picks up the importance weight

    log_importance_weight = leader_log_prob.detach() - follower_log_prob.detach()
    chosen = torch.randperm(leader_log_prob.shape[0], device = leader_log_prob.device)[:num_samples]

    return ppo_actor_loss(
        leader_log_prob[chosen],
        leader_log_prob.detach()[chosen] - log_importance_weight[chosen],
        follower_advantages[chosen],
        eps_clip = eps_clip,
        normalize_advantages = normalize_advantages
    ).mean()

# split and aggregate policy gradients (singla, agarwal, pathak - icml 2024)
# environments are split into blocks, each block running its own policy from a shared set of parameters
# the leader, block 0, additionally trains on importance weighted off policy data aggregated from all the followers
# the loop a training script needs is `sapg.update(sapg.collect(env, num_steps), opt, epochs = 6)`

class SAPG(nn.Module):
    def __init__(
        self,
        agent,
        num_blocks = 4,
        clip_range = 0.1,
        gamma = 0.99,
        gae_lambda = 0.95,
        ent_coef = 0.002,
        critic_coef = 0.5,
        value_clip_range: float | None = None,
        normalize_advantages = True,
        max_grad_norm = 1.
    ):
        super().__init__()
        self.agent = agent
        self.num_blocks = num_blocks
        self.clip_range = clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.critic_coef = critic_coef
        self.value_clip_range = value_clip_range
        self.normalize_advantages = normalize_advantages
        self.max_grad_norm = max_grad_norm

        # where a continuing rollout picks up

        self.last_obs = None

    def block_ids_for_envs(self, num_envs):
        assert divisible_by(num_envs, self.num_blocks), f'{num_envs} environments cannot be evenly split into {self.num_blocks} blocks'
        return repeat(torch.arange(self.num_blocks), 'b -> (b e)', e = num_envs // self.num_blocks)

    # gradients are handed to the optimizer averaged, the statistics that follow the population
    # as a batchnorm would - both no-ops when the process group is not up

    def sync_gradients(self, world = None):
        all_reduce_mean_([param.grad for param in self.agent.parameters()], world = world)

    def sync_buffers(self, world = None):
        all_reduce_mean_([
            buffer
            for module in self.agent.modules() if getattr(module, 'follows_population', False)
            for buffer in module.buffers()
        ], world = world)

    def loss(self, experience):
        # one pass over the rollout, the blocks split, the leader aggregating the followers
        # `update` sequences the passes - reach for this only to write your own loop

        experience.validate(num_blocks = self.num_blocks)

        agent, num_blocks = self.agent, self.num_blocks

        obs, block_ids = experience.obs, experience.block_ids
        time = experience.rewards.shape[-1]

        # advantages - the bootstrap is cut at termination, the recurrence at any done

        terminated, truncated = experience.terminated, experience.truncated

        if not exists(terminated) and not exists(truncated):
            # a rollout carrying no boundaries is taken as one continuing episode

            terminated = torch.zeros_like(experience.rewards, dtype = torch.bool)

        if exists(truncated):
            terminate_masks, done_masks = dones_to_masks(terminated = terminated, truncated = truncated)
        else:
            terminate_masks, done_masks = dones_to_masks(terminated = terminated), None

        gae = calc_gae(
            experience.rewards,
            experience.values,
            masks = terminate_masks,
            done_masks = done_masks,
            gamma = self.gamma,
            lam = self.gae_lambda,
            return_advantages = True
        )

        advantages, returns = gae.advantages, gae.returns

        # every block under its own policy, on whatever else the environment dealt the agent

        aux = default(experience.aux, dict())

        log_prob, value, entropy, *_ = agent.evaluate_actions(obs, experience.actions, priv_obs = experience.priv_obs, block_ids = block_ids, **aux)

        # on policy surrogate for every block, with the followers additionally maximizing entropy at increasing scales

        policy_loss = per_block_ppo_loss(
            log_prob,
            experience.log_probs,
            advantages,
            block_ids,
            num_blocks,
            eps_clip = self.clip_range,
            normalize_advantages = self.normalize_advantages
        ).mean()

        per_block_entropy = block_entropy(entropy, block_ids, num_blocks)

        entropy_bonus = sum(self.ent_coef * block_index * per_block_entropy[block_index] for block_index in range(1, num_blocks))

        # the leader aggregates the off policy data of all the followers

        off_policy_loss = obs.new_zeros(())

        followers = block_ids != 0

        if followers.any():
            follower_obs = obs[followers]
            follower_priv_obs = experience.priv_obs[followers] if exists(experience.priv_obs) else None
            follower_aux = {key: value[followers] for key, value in aux.items()}

            leader_log_prob, *_ = agent.evaluate_actions(follower_obs, experience.actions[followers], priv_obs = follower_priv_obs, block_ids = 0, **follower_aux)

            num_off_policy = min(follower_obs.shape[0] * time, int((~followers).sum()) * time)

            off_policy_loss = leader_off_policy_loss(
                rearrange(leader_log_prob, 'e t -> (e t)'),
                rearrange(experience.log_probs[followers], 'e t -> (e t)'),
                rearrange(advantages[followers], 'e t -> (e t)'),
                num_off_policy,
                eps_clip = self.clip_range,
                normalize_advantages = self.normalize_advantages
            )

        # shared critic over all of the data

        value_loss = clipped_value_loss_fn(
            value,
            experience.values[..., :-1],
            returns,
            clip = self.value_clip_range
        )

        loss = policy_loss + off_policy_loss + self.critic_coef * value_loss - entropy_bonus

        return SAPGLoss(loss, policy_loss, off_policy_loss, value_loss, per_block_entropy)

    def update(self, experience, opt, epochs = 6, world = None):
        # the whole update - `epochs` passes over the rollout, each backward averaged over the population,
        # clipped, and handed to the optimizer
        # a single pass per rollout leaves the critic too far behind to keep the agents learning, so the
        # passes live here, where a training script cannot silently skip them

        assert epochs >= 1, f'epochs must be positive, got {epochs}'

        for _ in range(epochs):
            out = self.loss(experience)

            opt.zero_grad()
            out.loss.backward()

            self.sync_gradients(world)
            clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            opt.step()

        self.sync_buffers(world)

        return out

    @torch.no_grad()
    def collect(
        self,
        env,
        num_steps,
        obs_key = 'obs',
        priv_obs_key = 'priv_obs',
        hidden = None,
        reset = True
    ):
        # roll out the split policies against any environment wrapped by env-ssl-wrapper
        # `reset` starts the worlds afresh - otherwise the rollout continues where the last one left off

        assert num_steps >= 1, f'num_steps must be positive, got {num_steps}'

        agent, device = self.agent, self.agent.device

        def cast_obs(obs):
            # anything beyond the observation and the privileged state - the scene, the past action - goes to the agent as is

            def to_tensor(value):
                return cast_tensor(value, device = device) if exists(value) else None

            if not isinstance(obs, dict):
                return to_tensor(obs), None, dict()

            aux = {key: to_tensor(value) for key, value in obs.items() if key not in (obs_key, priv_obs_key) and exists(value)}
            return to_tensor(obs[obs_key]), to_tensor(obs.get(priv_obs_key)), aux

        if reset or not exists(self.last_obs):
            obs, _ = env.reset()
            self.last_obs = cast_obs(obs)

        main_obs, priv_obs, aux = self.last_obs

        num_envs = main_obs.shape[0]
        block_ids = self.block_ids_for_envs(num_envs).to(device)

        obs_list = []
        priv_obs_list = []
        aux_list = []
        action_list = []
        log_prob_list = []
        value_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []

        for _ in range(num_steps):
            # recurrent agents take a hidden state in, and hand the next one back

            hidden_kwarg = dict(hidden = hidden) if exists(hidden) else dict()

            action, log_prob, value, *rest = agent.act(main_obs, priv_obs, block_ids = block_ids, **hidden_kwarg, **aux)
            hidden = rest[0] if rest else None

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_main_obs, next_priv_obs, next_aux = cast_obs(next_obs)

            terminated = cast_tensor(terminated, device = device, dtype = torch.bool)
            truncated = cast_tensor(truncated, device = device, dtype = torch.bool)

            obs_list.append(main_obs)
            priv_obs_list.append(priv_obs)
            aux_list.append(aux)
            action_list.append(action)
            log_prob_list.append(log_prob)
            value_list.append(value)
            reward_list.append(cast_tensor(reward, device = device))
            terminated_list.append(terminated)
            truncated_list.append(truncated)

            # reset the recurrent state of environments that finished

            done = terminated | truncated

            if exists(hidden) and done.any():
                hidden_state, hidden_cell = hidden
                keep = rearrange(~done, 'e -> 1 e 1')

                hidden = (hidden_state * keep, hidden_cell * keep)

            main_obs, priv_obs, aux = next_main_obs, next_priv_obs, next_aux

        self.last_obs = (main_obs, priv_obs, aux)

        # bootstrap the last value - agents that can give it without sampling do so

        if hasattr(agent, 'value'):
            last_value = agent.value(main_obs, priv_obs, **aux)
        else:
            _, last_value, *_ = agent.forward(main_obs, priv_obs, block_ids = block_ids, **hidden_kwarg, **aux)

        return Experience(
            obs = stack_time(obs_list),
            actions = stack_time(action_list),
            log_probs = stack_time(log_prob_list),
            values = torch.cat((stack_time(value_list), last_value.unsqueeze(-1)), dim = -1),
            rewards = stack_time(reward_list),
            block_ids = block_ids,
            priv_obs = stack_time(priv_obs_list) if exists(priv_obs_list[0]) else None,
            aux = {key: stack_time([step[key] for step in aux_list]) for key in aux} if aux else None,
            terminated = stack_time(terminated_list),
            truncated = stack_time(truncated_list)
        )
