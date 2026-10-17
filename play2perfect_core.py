"""Play2Perfect pick task on top of the play environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import warp as wp
from PIL import Image, ImageDraw

from play2perfect_env import (
    PICK_CONFIG,
    TASKS,
    AssetRegistry,
    Config,
    DwellSuccess,
    Entity,
    MjWarpEnv,
    PlayTask,
    Scene,
    Tool,
    ToolSampler,
    build_play_scene,
    project_path,
    quat_error,
    render,
    vec,
    yaw_quat,
)

ROOT = project_path(PICK_CONFIG.assets.root)
_ASSETS = AssetRegistry.from_config(PICK_CONFIG.assets)
ASSETS = {
    name: _ASSETS.path(name)
    for name in ("robot", "table", "pipette_raw", "pipette_body", "pipette_tube", "beam0", "beam2", "beam6")
}


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else (d() if callable(d) else d)

def asset(name: str) -> Path:
    return _ASSETS.path(name)


@dataclass(frozen = True)
class Hand:
    joints: tuple
    open_q: np.ndarray
    power_q: np.ndarray
    pinch_q: np.ndarray

    def q(self, name: str = "open"):
        return getattr(self, f"{name}_q")


@dataclass(frozen = True)
class Robot:
    name: str
    urdf: Path
    arm_joints: tuple
    hand: Hand
    default_q: np.ndarray


@dataclass(frozen = True)
class PickTool:
    name: str
    raw: Tool
    width: float
    goal_xml: str


def sharpa(config: Config | None = None) -> Hand:
    config = default(config, PICK_CONFIG)
    names = tuple(config.robot.hand.joints)
    open_q = np.zeros(len(names))
    power_q = np.zeros(len(names))
    pinch_q = np.zeros(len(names))
    for index, name in enumerate(names):
        if name.endswith(("_FE", "_PIP", "_DIP")):
            power_q[index] = float(config.hand.flex)
        power_q[index] = float(config.hand.power.get(name, power_q[index]))
        pinch_q[index] = float(config.hand.pinch.get(name, 0.0))
    return Hand(names, open_q, power_q, pinch_q)


def iiwa(config: Config | None = None, assets: AssetRegistry | None = None) -> Robot:
    config = default(config, PICK_CONFIG)
    assets = default(assets, _ASSETS)
    hand = sharpa(config)
    q = np.array(config.robot.arm.default_q, float)
    full_q = np.zeros(len(config.robot.arm.joints) + len(hand.joints))
    full_q[: len(q)] = q
    full_q[len(q):] = hand.open_q
    return Robot(config.robot.name, assets.path("robot"), tuple(config.robot.arm.joints), hand, full_q)


def make_tool(name: str | None = None, config: Config | None = None, assets: AssetRegistry | None = None) -> PickTool:
    config = default(config, PICK_CONFIG)
    assets = default(assets, _ASSETS)
    name = default(name, config.task.pick.default_tool)
    sampler = ToolSampler(config.tools, assets)
    raw = sampler.sample(num_per_type = 1, types = (name,), shuffle = False)[0]
    rgba = vec(config.scene.goal.ghost_rgba)

    if raw.mesh_parts:
        ghost = "\n".join(
            render(
                "pick_ghost_mesh.xml",
                index = index,
                mesh = mesh,
                scale = vec((scale, scale, scale)),
                rgba = rgba,
            )
            for index, (mesh, _, scale) in enumerate(raw.mesh_parts)
        )
    else:
        ghost = render(
            "pick_ghost_capsule.xml",
            radius = f"{raw.handle[1] / 2:.5f}",
            half_length = f"{raw.handle[0] / 2:.5f}",
            quat = config.geometry.cylinder_quat,
            rgba = rgba,
        )
    return PickTool(name, raw, float(raw.handle[1]), ghost)


# scene

def shelf_entity(config: Config | None = None) -> Entity:
    config = default(config, PICK_CONFIG)
    shelf = config.scene.shelf
    posts = "\n".join(
        render(
            "pick_shelf_post.xml",
            index = index,
            half = vec(shelf.post_half),
            pos = vec(
                (
                    float(shelf.x) + float(shelf.post_dx),
                    float(shelf.y) + float(dy),
                    float(shelf.top) + float(shelf.post_dz),
                )
            ),
            rgba = vec(shelf.rgba),
        )
        for index, dy in enumerate(shelf.post_dy)
    )
    xml = render(
        "pick_shelf.xml",
        pos = vec((float(shelf.x), float(shelf.y), float(shelf.top) - float(shelf.half[2]))),
        half = vec(shelf.half),
        rgba = vec(shelf.top_rgba),
        friction = vec(shelf.friction),
        posts = posts,
    )
    return Entity("shelf", xml)


class PickScene(Scene):
    def __init__(
        self,
        config: Config,
        assets: AssetRegistry,
        robot: Robot,
        tool: PickTool,
        camera = None,
        with_shelf: bool = True,
    ):
        super().__init__(config.scene.model, assets, config.simulation)
        self.robot = robot
        self.tool = tool
        self.camera = default(camera, (config.scene.camera.pos, config.scene.camera.lookat))
        extra = (shelf_entity(config),) if with_shelf else ()
        build_play_scene(
            config,
            assets,
            tool.raw,
            camera = self.camera,
            goal_xml = self.goal_xml(config),
            extra_entities = extra,
            scene = self,
        )

    def goal_xml(self, config: Config | None = None) -> str:
        config = default(config, PICK_CONFIG)
        return render(
            "pick_goal.xml",
            pos = vec(config.scene.goal.pos),
            gravcomp = config.scene.goal.gravcomp,
            ghost = self.tool.goal_xml,
        )


def scene(tool_name: str | None = None, with_shelf: bool | None = None, camera = None) -> PickScene:
    config = PICK_CONFIG
    assets = _ASSETS
    robot = iiwa(config, assets)
    tool = make_tool(tool_name, config, assets)
    with_shelf = bool(default(with_shelf, config.scene.with_shelf))
    return PickScene(config, assets, robot, tool, camera = camera, with_shelf = with_shelf)


# pick task

class PickTaskImpl(PlayTask):
    def __init__(
        self,
        config: Config,
        assets: AssetRegistry,
        scene_: PickScene,
        goal_tier: int | None = None,
    ):
        super().__init__(config, assets, tools = [scene_.tool.raw], scene = scene_)
        self.pick = config.task.pick
        self.goal_tier = int(default(goal_tier, self.pick.goal_tier))
        self.challenge = None
        self.pick_success = DwellSuccess(float(self.pick.position_tolerance), int(self.pick.dwell_steps))

    def bind(self, env) -> None:
        super().bind(env)
        self.pick_success.reset(env.nworld)
        self.prev_distance = np.zeros(env.nworld)
        self.lift = np.zeros(env.nworld)
        self.table_top = float(self.config.scene.table.z) + float(self.config.scene.table.size[2]) / 2

    def reset(self, env, worlds: np.ndarray) -> None:
        qpos = env.d.qpos.numpy()
        qvel = env.d.qvel.numpy()
        challenge = self.challenge
        object_xy = tuple(default(getattr(challenge, "object_xy", None), self.pick.start_xy))
        object_yaw = float(default(getattr(challenge, "object_yaw", None), self.pick.yaw))
        goal_tier = int(default(getattr(challenge, "goal_tier", None), self.goal_tier))
        gap = float(self.pick.object_gap)

        qpos[np.ix_(worlds, self.qadr)] = self.default_joint_pos
        qvel[np.ix_(worlds, self.qadr)] = 0.0
        self.targets[worlds] = self.default_joint_pos
        self.prev_targets[worlds] = self.default_joint_pos

        half_height = self._object_rest_half_height(env, worlds)
        object_z = self.table_top + half_height + gap
        qpos[np.ix_(worlds, np.arange(self.object_free, self.object_free + 3))] = np.stack(
            [
                np.full(len(worlds), object_xy[0]),
                np.full(len(worlds), object_xy[1]),
                object_z,
            ],
            axis = 1,
        )
        qpos[np.ix_(worlds, np.arange(self.object_free + 3, self.object_free + 7))] = yaw_quat(object_yaw)
        qvel[np.ix_(worlds, np.arange(self.object_dofadr, self.object_dofadr + 6))] = 0.0
        self.object_init_z[worlds] = object_z

        base_xy = self.pick.shelf_goal if goal_tier else self.pick.table_goal
        base_z = float(self.config.scene.shelf.top) + half_height if goal_tier else self.table_top + half_height
        goal = np.stack(
            [
                np.full(len(worlds), float(base_xy[0])),
                np.full(len(worlds), float(base_xy[1])),
                base_z,
            ],
            axis = 1,
        )
        goal_quat = yaw_quat(float(self.pick.yaw))
        env.body_pos[worlds, self.goal_bid] = goal
        env.body_quat[worlds, self.goal_bid] = goal_quat
        self.goal.goal_pose[worlds] = np.concatenate([goal, np.tile(goal_quat, (len(worlds), 1))], axis = 1)

        self.pick_success.dwell[worlds] = 0
        self.successes[worlds] = 0
        self.episode_len[worlds] = -1  # update() below advances to 0
        self.closest_kp[worlds] = -1.0
        self.closest_ft[worlds] = -1.0
        self.done[worlds] = False
        self.prev_distance[worlds] = 0.0
        self.lift[worlds] = 0.0

        env.d.qpos = wp.array(qpos, dtype = wp.float32)
        env.d.qvel = wp.array(qvel, dtype = wp.float32)
        env.d.ctrl = wp.array(self.targets, dtype = wp.float32)
        env._upload_world_arrays()
        mjw.forward(env.m, env.d)
        self.update(env)

    def update(self, env) -> None:
        self.episode_len += 1
        self._update_kinematics(env)

        object_pos = self.object_pos
        goal_pos = self.goal_pos
        position_error = np.linalg.norm(object_pos - goal_pos, axis = -1)
        rotation_error = np.array([
            quat_error(quat, target) for quat, target in zip(self.object_quat, self.goal_quat)
        ])
        near = (position_error < float(self.pick.position_tolerance)) & (
            rotation_error < float(self.pick.rotation_tolerance)
        )
        dwell = np.where(near, self.pick_success.dwell + 1, 0)
        success = dwell >= int(self.pick.dwell_steps)
        self.pick_success.dwell = dwell

        self.near = near
        self.success = success
        self.dwell = dwell
        self.distance = position_error
        self.keypoint_dist = position_error
        self.lift = object_pos[:, 2] - self.table_top
        self.reward_value, self.last_reward_terms = self.reward.compute(env)

        termination = self.config.task.termination
        fall = object_pos[:, 2] < float(termination.object_fall_z)
        timeout = self.episode_len >= int(self.config.task.episode.length)
        self.done = success | fall | timeout
        env.done = self.done
        env.success = success
        env.reward = self.reward_value
        env.last_reward_terms = self.last_reward_terms
        env.info = {
            "success": near,
            "is_success": success,
            "dwell": dwell,
            "distance": position_error,
            "rotation_error": rotation_error,
            "lift": self.lift,
        }
        env.terminated = np.zeros(env.nworld, bool)
        env.truncated = np.zeros(env.nworld, bool)
        self.prev_targets = self.targets.copy()

    def observe(self, env) -> dict:
        obs = super().observe(env)
        obs.update(
            object_pos = self.object_pos,
            goal_pos = self.goal_pos,
            object_pos_rel_goal = self.object_pos - self.goal_pos,
            object_quat = self.object_quat,
            goal_quat = self.goal_quat,
            lift = self.lift,
        )
        return obs


class PickTask:
    """User-facing pick task with ``reset / step / observe / render``."""

    def __init__(
        self,
        scene_: PickScene | None = None,
        nworld: int | None = None,
        seed: int | None = None,
        goal_tier: int | None = None,
        verbose: bool = False,
        config: Config | None = None,
        assets: AssetRegistry | None = None,
    ):
        config = default(config, PICK_CONFIG)
        assets = default(assets, _ASSETS)
        self.config = config
        self.scene = default(scene_, lambda: scene())
        self.impl = PickTaskImpl(config, assets, self.scene, goal_tier = goal_tier)
        self.env = MjWarpEnv(self.impl, nworld = nworld, seed = seed, verbose = verbose, config = config)
        self.action_space = self.impl.action_space

    def reset(self, challenge = None):
        self.impl.challenge = challenge
        return self.env.reset()

    def observe(self):
        return self.env.observe()

    def step(self, action):
        obs, reward, _, _, _ = self.env.step(action)
        return obs, reward, self.env.done | self.env.success, self.env.success

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()

    @property
    def nworld(self) -> int:
        return self.env.nworld

    @property
    def done(self) -> np.ndarray:
        return self.env.done

@dataclass(frozen = True)
class Sim2Real:
    friction: float = float(PICK_CONFIG.task.sim2real.friction)
    mass: float = float(PICK_CONFIG.task.sim2real.mass)
    action_delay: int = int(PICK_CONFIG.task.sim2real.action_delay)

    def apply(self, env, rng):
        friction = 1.0 + self.friction * rng.uniform(-1.0, 1.0, (env.nworld, 1, 1))
        mass = 1.0 + self.mass * rng.uniform(-1.0, 1.0, env.nworld)
        env.m.geom_friction.assign((env.m.geom_friction.numpy() * friction).astype(np.float32))
        body_mass = env.m.body_mass.numpy()
        body_mass[:, env._obj_bid] *= mass
        env.m.body_mass.assign(body_mass.astype(np.float32))
        return friction, mass


@TASKS.register("pick")
def _make_pick(**kwargs):
    return PickTask(**kwargs)

@dataclass(frozen = True)
class Stage:
    name: str
    preshape: str


STAGES = tuple(Stage(**dict(item)) for item in PICK_CONFIG.task.stages)


@dataclass(frozen = True)
class Challenge:
    stage: str
    object_xy: tuple
    object_yaw: float
    goal_tier: int
    preshape: str
    difficulty: float


_CHALLENGE = PICK_CONFIG.task.challenge


@dataclass(frozen = True)
class ChallengeSpace:
    stage: int = int(_CHALLENGE.stage)
    goal_tier: int = int(_CHALLENGE.goal_tier)
    object_region: tuple = tuple(_CHALLENGE.object_region)
    yaw_range: float = float(_CHALLENGE.yaw_range)

    def difficulty(self) -> float:
        weights = _CHALLENGE.difficulty
        return (
            (len(STAGES) - 1 - self.stage) * float(weights.stage)
            + float(weights.goal_tier) * self.goal_tier
            + float(weights.yaw) * self.yaw_range
        )

    def __call__(self, rng, idx: int = 0, step: int = 0):
        x0, x1, y0, y1 = self.object_region
        xy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        yaw = math.pi / 2.0 + float(rng.uniform(-self.yaw_range, self.yaw_range))
        stage = STAGES[self.stage]
        return Challenge(stage.name, xy, yaw, self.goal_tier, stage.preshape, self.difficulty())


def bank(goal_tiers = (0,), stages = None):
    stages = default(stages, range(len(STAGES)))
    return [ChallengeSpace(stage, tier) for tier in goal_tiers for stage in stages]


def harden(space: ChallengeSpace, rng) -> ChallengeSpace:
    config = _CHALLENGE.harden
    return replace(
        space,
        stage = max(0, space.stage + int(config.stage_step)),
        goal_tier = min(
            int(_CHALLENGE.goal_tier_max),
            space.goal_tier + (space.stage == int(config.promote_goal_stage)),
        ),
        yaw_range = min(float(config.yaw_max), space.yaw_range * float(config.yaw_scale)),
    )


def soften(space: ChallengeSpace, rng) -> ChallengeSpace:
    config = _CHALLENGE.soften
    return replace(
        space,
        stage = min(len(STAGES) - 1, space.stage + int(config.stage_step)),
        yaw_range = max(float(config.yaw_min), space.yaw_range * float(config.yaw_scale)),
    )


def challenges(goal_tiers = (0, 1)):
    for space in bank(goal_tiers):
        print(space.stage, space.goal_tier, round(space.difficulty(), 1))

_RECORD = PICK_CONFIG.record


def record(task, policy, out = "out/core/rollout", steps = None, every = None, fps = None):
    steps = int(default(steps, _RECORD.steps))
    every = int(default(every, _RECORD.frame_every))
    fps = int(default(fps, _RECORD.fps))
    out, frames = Path(out), []
    obs = task.observe()
    for step in range(steps):
        obs, _, done, success = task.step(policy(obs))
        if step % every == 0 or step == steps - 1:
            image = Image.fromarray((np.clip(task.render()[0], 0.0, 1.0) * 255).astype(np.uint8))
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rectangle((0, 0, image.width, int(_RECORD.footer_height)), fill = tuple(_RECORD.overlay_rgba))
            offset = _RECORD.text_offset
            draw.text(
                (offset[0], offset[1]),
                f"step {step:03d}  success {bool(success[0])}",
                fill = tuple(_RECORD.text_rgba),
            )
            draw.text(
                (offset[0], offset[1] + int(_RECORD.text_line_height)),
                f"object z {obs['object_pos'][0, 2]:.3f}",
                fill = tuple(_RECORD.secondary_text_rgba),
            )
            frames.append(image)
        if done.any():
            break
    out.parent.mkdir(parents = True, exist_ok = True)
    frames[0].save(
        out.with_suffix(".gif"),
        save_all = True,
        append_images = frames[1:],
        duration = int(1000 / fps),
        loop = 0,
    )
    return out.with_suffix(".gif")


def smoke(record_video: bool = False):
    scene_ = scene()
    task = PickTask(scene_, verbose = True)
    print("assets", sorted(ASSETS))
    print(
        "robot", scene_.robot.name,
        "| hand", len(scene_.robot.hand.joints),
        "| tool", scene_.tool.name,
        "| mesh", bool(scene_.tool.raw.mesh),
    )
    print("scene", [entity.name for entity in scene_.entities])
    obs = task.reset()
    print("object", np.round(obs["object_pos"], 3), "goal", np.round(obs["goal_pos"], 3))
    for _ in range(int(_RECORD.smoke_steps)):
        obs, reward, done, success = task.step(np.zeros((1, len(scene_.robot.default_q)), np.float32))
    print("reward", float(reward[0]), "success", bool(success[0]), "done", bool(done[0]))
    if record_video:
        print(record(task, lambda obs: np.zeros((1, len(scene_.robot.default_q)), np.float32), "out/core/smoke"))
    task.close()


def demo(tool = None, goal_tier = None, steps = None, out = None):
    tool = default(tool, PICK_CONFIG.task.pick.default_tool)
    goal_tier = int(default(goal_tier, PICK_CONFIG.task.pick.goal_tier))
    scene_ = scene(tool)
    task = PickTask(scene_, goal_tier = goal_tier, verbose = True)
    rng = np.random.default_rng(0)
    policy = lambda obs: rng.uniform(-1.0, 1.0, (1, len(scene_.robot.default_q)))
    out = default(out, f"out/core/{tool}_{goal_tier}")
    gif = record(task, policy, out, steps = steps)
    print(gif)
    task.close()
    return gif


if __name__ == "__main__":
    from fire import Fire

    Fire({"smoke": smoke, "demo": demo, "challenges": challenges})
