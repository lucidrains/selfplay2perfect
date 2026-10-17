## selfplay2perfect

Implementation of [Play2Perfect](https://play2perfect.github.io/) with a
success-guided self-play layer, plus a small config-driven simulation library
built on MuJoCo-Warp.

## Simulation library

The working implementation stays in the two original top-level modules,
organized by section. All data lives in one config file:

| file | what it is |
|---|---|
| `play2perfect_env.py` | framework + play environment: config, registries, assets, scene, tools, rewards, goals, task protocol, vectorized env, play task |
| `play2perfect_core.py` | pick task on the same robot/tool stack: scene, pick task, challenge space, sim2real, recorder |
| `configs/play2perfect.yaml` | every tunable number and every MJCF template |

The extension points are ordinary classes and functions in
`play2perfect_env.py`.

## Configuration

`configs/play2perfect.yaml` has three sections:

- `play` - the base environment: robot, tools, simulation, rewards, reset
- `pick` - a sparse override merged over `play` at load time
- `templates` - all MJCF, as `$name` templates

```python
import play2perfect_env as E

config = E.load_config()
play, pick = config.play, config.pick
xml = E.build_scene_xml()
```

## Designing a new environment

The simulator loop is generic. Environment-specific behaviour is composed from
the pieces below:

- **assets** - register a path, kind and name prefix in the YAML
  `assets.registry`; consume it with `AssetRegistry.path` or `Scene.add_urdf`.
- **scenes** - compose `Entity` XML with `Scene.add_entity`; the base MJCF
  skeleton is the `base.xml` template in the config.
- **goals** - implement a goal class with `reset` / `update` (see
  `KeypointGoal`) and register it with `SCENARIOS.register`.
- **rewards** - subclass `RewardTerm`, register it, and add it to a YAML
  `reward.terms` list:

  ```python
  @E.REWARDS.register("my_bonus")
  class MyBonus(E.RewardTerm):
      def value(self, env):
          return env.task.my_signal * self.params["scale"]
  ```

- **tasks** - subclass `Task`, implement `bind`, `reset`, `apply_action`,
  `update` and `observe`, then pass it to `MjWarpEnv`.
- **environments** - register a constructor with `TASKS.register` and use
  `E.make("name", ...)`.

## Run

```bash
.venv/bin/python play2perfect_core.py smoke
.venv/bin/python play2perfect_core.py demo --tool pipette --goal_tier 0
.venv/bin/python play2perfect_core.py demo --tool hammer --goal_tier 1
.venv/bin/python play2perfect_env.py demo --nworld 4 --steps 60 --out out/play
.venv/bin/python -m pytest tests
```

`demo` records a real MuJoCo rollout (GIF) under `out/core/`. `smoke` compiles
the paper robot + pipette + shelf and steps the task.

## Assets

All assets are under `assets/play2perfect/` (see its README). Source only;
generated MJCF lives in `.cache/`. The real pipette is the cooked AutoBio OBJ,
not a primitive.

## Codebase

- `selfplay2perfect/` — training/curriculum core: `ActorCritic` (Simba-style
  shared encoder), `SAPG` (single-network PPO when `num_blocks=1`), SGS sampler,
  self-certified adversary, success criteria, metrics.
- `play2perfect_env.py` — the simulation framework and play environment.
- `play2perfect_core.py` — pick task, challenge space, sim2real and recording.
- `configs/play2perfect.yaml` — all simulation/task numbers and MJCF templates.
- `tests/` — core library tests; `tests/test_core.py` checks assets, scene and
  task; `tests/test_sim_library.py` checks config, templates, assets and reward
  registration.
- `train_pusht_*.py`, `report_*.py`, `probe_pusht_collapse.py` — the 2-D SGS
  testbed and reports from `EXPERIMENTS.md`, independent of the robot env.
- `runs/`, `out/` — experiment outputs. `out/real/` holds the real-robot
  failure videos and scenes.

## Sim2real

The paper URDF/STL, joint gains/damping/armature/friction and the real pipette
OBJ are used directly. The table and shelf are the only added scene geometry.
Collision uses the paper's convex hulls/proxies; contact parameters are MuJoCo
defaults. `play2perfect_core.Sim2Real` is the hook for friction/mass/actuator
randomization. The MLX engine in the lucidrains fork covers plane/sphere/capsule
scenes; the Sharpa's STL collision meshes require the Warp backend.

## Citations

```bibtex
@misc{lum2026play2perfectmattersdexterousplay,
    title   = {Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?},
    author  = {Tyler Ga Wei Lum and Kushal Kedia and C. Karen Liu and Jeannette Bohg},
    year    = {2026},
    eprint  = {2606.26428},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO},
    url     = {https://arxiv.org/abs/2606.26428}
}
```

```bibtex
@misc{zhang2026alphanistemergentassemblyskills,
    title     = {AlphaNIST: Emergent Assembly Skills from Binary Task Rewards},
    author    = {Octi Zhang and Iretiayo Akinola and Rosario Scalise and Mateo Guaman Castro and Patrick Yin and Tobias Widmer and Miles Macklin and Chris Amevor and Dylan Turpin and Dirk Van Gelder and Jan Carius and Ankur Handa and Gavriel State and Yunrong Guo and Yashraj Narang},
    year      = {2026},
    eprint    = {2610.13354},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url       = {https://arxiv.org/abs/2610.13354}
}
```
