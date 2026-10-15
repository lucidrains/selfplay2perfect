from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, Parameter

from einops import reduce, rearrange

from torch_einops_utils import pad_right_ndim_to
from torch_einops_utils.shape import assert_shape, shape

from x_mlps_pytorch import MLP, Feedforwards, ContinuousPastActions
from mean_conc_beta import Beta

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# running normalization - the RN of the figure, the statistics tracked on the rollouts alone,
# so that the loss pass stays a pure function of the observations

class RunningNorm(Module):
    # the statistics follow the whole population when the population is split across nodes

    follows_population = True

    def __init__(
        self,
        dim,
        momentum = 0.99,
        eps = 1e-5
    ):
        super().__init__()
        self.momentum, self.eps = momentum, eps

        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))

    def forward(self, x):
        if self.training and not torch.is_grad_enabled():
            batch = x.reshape(-1, x.shape[-1])
            self.mean.lerp_(batch.mean(dim = 0), 1. - self.momentum)
            self.var.lerp_(batch.var(dim = 0), 1. - self.momentum)

        return (x - self.mean) * (self.var + self.eps).rsqrt()

# the scene encoder of the paper - a two layer elu encoder per point, max pooled over the points

class PointCloudEncoder(Module):
    def __init__(
        self,
        dim_point = 3,
        dim_hidden = 256,
        dim_embed = 128,
        activation = nn.ELU()
    ):
        super().__init__()
        self.proj = MLP(dim_point, dim_hidden, dim_embed, activation = activation)

    def forward(self, points):
        return reduce(self.proj(points), '... n d -> ... d', 'max')

# actor critic from the alphanist paper - https://arxiv.org/abs/2610.13354
# proprioception with the last action, and the scene - points by default, any encoder can be swapped in -
# are encoded apart, fused, then read out by a tower each for the actor and the critic
# the actor gives the mean and concentration of a beta over the action bounds, the critic a value

class ActorCritic(Module):
    def __init__(
        self,
        dim_obs,
        dim_action,
        dim_priv_obs = None,
        num_blocks = 1,
        dim_proprio_embed = 150,
        dim_scene_embed = 128,
        scene_encoder = None,
        dim_embed = 256,
        depth = 2,
        expansion_factor = 4.,
        activation = nn.SiLU(),
        beta = None
    ):
        super().__init__()
        self.dim_obs, self.dim_action, self.num_blocks = dim_obs, dim_action, num_blocks
        self.dim_priv_obs = default(dim_priv_obs, 0)
        self.dim_scene_embed = dim_scene_embed

        # the last action given to the environment, flagged absent on the first step of an episode

        self.past_action_encoder = ContinuousPastActions(dim_action)

        # proprioception, running normalized, then the scene, as in the figure

        dim_proprio = dim_obs + self.dim_priv_obs + dim_action + 1

        self.proprio_norm = RunningNorm(dim_proprio)
        self.proprio_encoder = MLP(dim_proprio, dim_proprio_embed, dim_proprio_embed, activation = activation)
        self.scene_encoder = default(scene_encoder, PointCloudEncoder(dim_embed = dim_scene_embed))

        # fusion, then a tower each for the actor and the critic, with separate parameters

        tower_kwargs = dict(
            dim = dim_embed,
            depth = depth,
            dim_in = dim_proprio_embed + dim_scene_embed,
            activation = activation,
            expansion_factor = expansion_factor,
            final_norm = True
        )

        self.actor = Feedforwards(**tower_kwargs, dim_out = dim_action * 2)
        self.critic = Feedforwards(**tower_kwargs, dim_out = 1)

        # the action distribution over the bounds, and the per block exploration scale,
        # a temperature on the concentration - the inverse of a per block log std

        self.beta = default(beta, Beta())
        self.log_temperature = Parameter(torch.zeros(num_blocks))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def temperature(self):
        return self.log_temperature.clamp(-2., 2.).exp()

    def encode(self, obs, priv_obs = None, scene = None, past_action = None, has_past_actions = None):
        # zeros stand in for what the environment does not give - privileged state, scene

        if not exists(past_action) and not exists(has_past_actions):
            has_past_actions = obs.new_zeros(shape(obs, '... d').ellipsis, dtype = torch.bool)

        past_action = self.past_action_encoder(past_action, has_past_actions)
        priv_obs = default(priv_obs, obs.new_zeros(shape(obs, '... d').replace(d = self.dim_priv_obs)))

        proprio = self.proprio_encoder(self.proprio_norm(torch.cat((obs, priv_obs, past_action), dim = -1)))

        scene_embed = self.scene_encoder(scene) if exists(scene) else proprio.new_zeros(shape(proprio, '... d').replace(d = self.dim_scene_embed))

        return torch.cat((proprio, scene_embed), dim = -1)

    def forward(self, obs, priv_obs = None, scene = None, past_action = None, has_past_actions = None, block_ids = None):
        # obs - (..., dim_obs) proprioception, scene - (..., whatever the scene encoder takes)
        # returns the action distribution of the block policy, and the value

        assert_shape(obs, '... d', d = self.dim_obs)

        embed = self.encode(obs, priv_obs, scene, past_action, has_past_actions)

        action_params = rearrange(self.actor(embed), '... (a two) -> ... a two', two = 2)
        value = rearrange(self.critic(embed), '... 1 -> ...')

        # the mean / concentration beta over the action bounds, with the block temperature applied to the
        # concentration before the unimodal correction, as `Beta.forward` would

        unit_mean = self.beta.unit_mean(action_params)
        concentration = self.beta.concentration(action_params[..., 1], indexed = True)

        temperature = pad_right_ndim_to(self.temperature[default(block_ids, 0)], concentration.ndim)

        return self.beta.from_transformed((unit_mean, concentration / temperature), apply_unimodal = True), value

    def act(self, *args, **kwargs):
        dist, value = self(*args, **kwargs)
        action = dist.sample()

        return action, dist.log_prob(action).sum(dim = -1), value

    def value(self, *args, **kwargs):
        return rearrange(self.critic(self.encode(*args, **kwargs)), '... 1 -> ...')

    def evaluate_actions(self, obs, actions, **kwargs):
        dist, value = self(obs, **kwargs)

        return dist.log_prob(actions).sum(dim = -1), value, dist.entropy().sum(dim = -1)
