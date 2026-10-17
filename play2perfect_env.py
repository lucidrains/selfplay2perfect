"""Play2Perfect simulation library on MuJoCo-Warp."""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
import yaml
from env_ssl_wrapper import InferredSpace

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else (d() if callable(d) else d)

# config

PROJECT_ROOT = Path(os.environ.get("PLAY2PERFECT_ROOT", Path(__file__).resolve().parent)).resolve()
CONFIG_PATH = PROJECT_ROOT / "configs" / "play2perfect.yaml"


def project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _wrap(value: Any, path: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        return Config(value, path)
    if isinstance(value, list):
        return [_wrap(item, path + (str(index),)) for index, item in enumerate(value)]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return {key: _unwrap(value[key]) for key in value}
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config(Mapping):
    """Read-only mapping with ``config.robot.arm.joints`` access."""

    def __init__(self, data: Mapping, path: tuple[str, ...] = (), source: Path | None = None):
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_source", source)

    @property
    def source(self) -> Path | None:
        return self._source

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key], self._path + (str(key),))

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as error:
            dotted = ".".join(self._path + (key,))
            raise AttributeError(f"missing config key: {dotted}") from error

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self._data else default

    def as_dict(self) -> dict:
        return {key: _unwrap(self[key]) for key in self}

    def __repr__(self) -> str:
        dotted = ".".join(self._path) or "<root>"
        return f"Config({dotted}, keys={list(self._data)})"


@cache
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    source = Path(default(path, CONFIG_PATH)).expanduser()
    data = yaml.safe_load(source.read_text()) or {}
    if source.resolve() != CONFIG_PATH:
        defaults = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        data = deep_merge(defaults, data)
    play = data.get("play", data)
    pick = deep_merge(play, data.get("pick", {}))
    templates = data.get("templates", {})
    return Config({"play": play, "pick": pick, "templates": templates}, source = source)


CONFIG = load_config()
PLAY_CONFIG = CONFIG.play
PICK_CONFIG = CONFIG.pick
TEMPLATES = CONFIG.templates

ARM_JOINT_NAMES = tuple(PLAY_CONFIG.robot.arm.joints)
HAND_JOINT_NAMES = tuple(PLAY_CONFIG.robot.hand.joints)
JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES
ARM_DEFAULT_JOINT_POS = dict(zip(ARM_JOINT_NAMES, PLAY_CONFIG.robot.arm.default_q))
TOOL_TYPES = tuple(PLAY_CONFIG.tools.types)
DEFAULT_TOOL_TYPES = tuple(PLAY_CONFIG.tools.default_types)
ASSET_DIR = project_path(PLAY_CONFIG.assets.root)
TABLE_Z = float(PLAY_CONFIG.scene.table.z)
TABLE_HALF = float(PLAY_CONFIG.scene.table.size[2]) / 2
TABLE_TOP = TABLE_Z + TABLE_HALF
_CYL_QUAT = PLAY_CONFIG.geometry.cylinder_quat


@cache
def _template(name: str) -> Template:
    return Template(TEMPLATES[name])


def render(template_name: str, **values) -> str:
    return _template(template_name).substitute(values)


def vec(values, precision: int = 6) -> str:
    return " ".join(f"{float(value):.{precision}f}" for value in values)


# registries

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, key: str, item: Any = None):
        if exists(item):
            self._items[key] = item
            return item

        def decorator(obj: Any) -> Any:
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> Any:
        try:
            return self._items[key]
        except KeyError as error:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"unknown {self.name} '{key}' (available: {available})") from error

    def create(self, key: str, *args, **kwargs) -> Any:
        return self.get(key)(*args, **kwargs)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, {list(self.keys())})"


TASKS = Registry("task")
REWARDS = Registry("reward")
SCENARIOS = Registry("scenario")


# spatial math

def quat_rotate_wxyz(quat, vector) -> np.ndarray:
    w, x, y, z = quat
    vector = np.asarray(vector, float)
    axis = np.array([x, y, z])
    return vector + 2 * np.cross(axis, np.cross(axis, vector) + w * vector)


def quat_mul_wxyz(left, right) -> np.ndarray:
    aw, ax, ay, az = left
    bw, bx, by, bz = right
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def rand_quat_wxyz(count: int, rng) -> np.ndarray:
    u1, u2, u3 = rng.random(count), rng.random(count), rng.random(count)
    scale = np.sqrt(1.0 - u1)[:, None]
    return np.concatenate(
        [
            np.sqrt(u1)[:, None] * np.cos(2 * np.pi * u3)[:, None],
            scale * np.sin(2 * np.pi * u2)[:, None],
            scale * np.cos(2 * np.pi * u2)[:, None],
            np.sqrt(u1)[:, None] * np.sin(2 * np.pi * u3)[:, None],
        ],
        axis = 1,
    )[:, [3, 0, 1, 2]]


def yaw_quat_wxyz(count: int, rng) -> np.ndarray:
    yaw = rng.uniform(-np.pi, np.pi, count)
    return np.stack([np.cos(yaw / 2), np.zeros(count), np.zeros(count), np.sin(yaw / 2)], axis = 1)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def quat_error(quat, goal) -> float:
    return 2.0 * math.acos(float(np.clip(abs(np.dot(quat, goal)), -1.0, 1.0)))


def keypoints_world(center, quat, offsets) -> np.ndarray:
    rotated = np.array([quat_rotate_wxyz(q, offset) for q in quat for offset in offsets])
    return center[:, None, :] + rotated.reshape(len(center), len(offsets), 3)


def lookat_quat(pos, target) -> np.ndarray:
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    forward = target - pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, (0.0, 0.0, 1.0))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return mat_to_quat_wxyz(np.stack([right, up, -forward], axis = 1))


def mat_to_quat_wxyz(matrix) -> np.ndarray:
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w, x, y, z = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w, x, y, z = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w, x, y, z = (
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w, x, y, z = (
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        )
    return np.array([w, x, y, z])


# assets

_URDF_STRIP = re.compile(
    r"<sdf\b[^>]*/>|<self_collision_checking\b[^>]*/>|<safety_controller\b[^>]*/>"
    r"|<self_collision_checking\b.*?</self_collision_checking>",
    re.DOTALL,
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass(frozen = True)
class Asset:
    name: str
    path: Path
    kind: str = "urdf"
    prefix: str = ""


class AssetRegistry:
    """Named assets rooted at a project asset directory."""

    def __init__(self, root: str | Path, cache: str | Path):
        self.root = project_path(root)
        self.cache = project_path(cache)
        self._assets: dict[str, Asset] = {}

    @classmethod
    def from_config(cls, config: Config) -> AssetRegistry:
        registry = cls(config.root, config.cache)
        for name, spec in config.registry.items():
            registry.register(
                name,
                spec.path,
                kind = spec.get("kind", "urdf"),
                prefix = spec.get("prefix", ""),
            )
        return registry

    def register(
        self,
        name: str,
        path: str | Path,
        *,
        kind: str = "urdf",
        prefix: str = "",
    ) -> Asset:
        path = Path(path)
        asset = Asset(name, path if path.is_absolute() else (self.root / path), kind, prefix)
        self._assets[name] = asset
        return asset

    def get(self, name: str) -> Asset:
        try:
            return self._assets[name]
        except KeyError as error:
            available = ", ".join(sorted(self._assets)) or "<empty>"
            raise KeyError(f"unknown asset '{name}' (available: {available})") from error

    def path(self, name: str) -> Path:
        return self.get(name).path

    def __contains__(self, name: object) -> bool:
        return name in self._assets

    def urdf_fragment(self, name: str) -> tuple[str, str]:
        """Return ``(asset_xml, body_xml)`` for a registered URDF, cached on disk."""

        asset = self.get(name)
        if asset.kind != "urdf":
            raise TypeError(f"asset '{name}' is {asset.kind}, not urdf")

        path = Path(asset.path)
        self.cache.mkdir(parents = True, exist_ok = True)
        key = f"{asset.prefix}{re.sub(r'[^A-Za-z0-9_.-]', '_', path.stem)}"
        cache = self.cache / f"{key}.xml"

        stale = not cache.exists()
        if not stale:
            deps = [path] + [path.parent / f for f in re.findall(r'filename="([^"]+)"', path.read_text())]
            stamp = max(p.stat().st_mtime for p in deps if p.exists())
            stale = cache.stat().st_mtime < stamp

        if stale:
            text = _URDF_STRIP.sub("", path.read_text())
            text = re.sub(
                r'filename="([^"]+)"',
                lambda match: f'filename="{(path.parent / match.group(1)).resolve()}"'
                if not Path(match.group(1)).is_absolute()
                else match.group(0),
                text,
            )
            tmp = self.cache / f"{key}.urdf"
            tmp.write_text(text)
            mujoco.mj_saveLastXML(str(cache), mujoco.MjModel.from_xml_path(str(tmp)))
            tmp.unlink()

        root = ET.fromstring(cache.read_text())
        asset_root = ET.Element("asset")
        name_map = {}
        assets_src = next((element for element in root if _local_name(element.tag) == "asset"), None)
        for mesh in default(assets_src, []):
            if _local_name(mesh.tag) != "mesh":
                continue
            old, file = mesh.get("name"), mesh.get("file")
            name_map[old] = f"{asset.prefix}{old}"
            mesh.set("name", name_map[old])
            if file:
                mesh_path = Path(file)
                mesh.set("file", str(mesh_path if mesh_path.is_absolute() else (path.parent / mesh_path).resolve()))
            asset_root.append(mesh)

        worldbody = next(element for element in root if _local_name(element.tag) == "worldbody")
        for element in worldbody.iter():
            if element.get("mesh") in name_map:
                element.set("mesh", name_map[element.get("mesh")])
            if "name" in element.attrib:
                element.set("name", f"{asset.prefix}{element.get('name')}")

        assets_xml = "".join(ET.tostring(child, encoding = "unicode") for child in asset_root)
        bodies_xml = "".join(ET.tostring(child, encoding = "unicode") for child in worldbody)
        return assets_xml, bodies_xml


# scene

@dataclass(frozen = True)
class Entity:
    name: str
    xml: str


@dataclass(frozen = True)
class Camera:
    name: str
    pos: tuple
    lookat: tuple
    fovy: float = 45.0

    def xml(self) -> str:
        return render(
            "camera.xml",
            name = self.name,
            pos = vec(self.pos),
            quat = vec(lookat_quat(self.pos, self.lookat)),
            fovy = f"{float(self.fovy):.6f}",
        )


class Scene:
    """A MJCF scene assembled from registered assets and XML entities."""

    def __init__(self, model: str, assets: AssetRegistry, simulation: Config):
        self.model = model
        self.registry = assets
        self.simulation = simulation
        self.entities: list[Entity] = []
        self._asset_xml: list[str] = []
        self._actuator_xml: list[str] = []

    def add_asset_xml(self, xml: str) -> None:
        if xml.strip():
            self._asset_xml.append(xml.strip())

    def add_urdf(self, name: str) -> str:
        assets, bodies = self.registry.urdf_fragment(name)
        self.add_asset_xml(assets)
        return bodies

    def add_entity(self, name: str, xml: str) -> Entity:
        entity = Entity(name, xml.strip())
        self.entities.append(entity)
        return entity

    def add_camera(self, camera: Camera) -> Entity:
        return self.add_entity(camera.name, camera.xml())

    def add_actuator_xml(self, xml: str) -> None:
        if xml.strip():
            self._actuator_xml.append(xml.strip())

    def xml(self) -> str:
        simulation = self.simulation
        default_geom = simulation.default_geom
        return render(
            "base.xml",
            model = self.model,
            timestep = f"{1.0 / float(simulation.physics_hz):.8f}",
            integrator = simulation.integrator,
            cone = simulation.cone,
            solver = simulation.solver,
            iterations = simulation.iterations,
            tolerance = simulation.tolerance,
            impratio = simulation.impratio,
            solref = vec(default_geom.solref),
            solimp = vec(default_geom.solimp),
            default_friction = vec(default_geom.friction),
            assets = "\n".join(self._asset_xml),
            entities = "\n".join(entity.xml for entity in self.entities),
            actuators = "\n".join(self._actuator_xml),
        )


# tools

@dataclass(frozen = True)
class ToolDistribution:
    type: str
    handle_min: tuple
    handle_max: tuple
    head_min: tuple | None
    head_max: tuple | None
    handle_density: tuple
    head_density: tuple | None

    @property
    def shape(self) -> str:
        return "box" if len(self.handle_min) == 3 else "cylinder"


@dataclass
class Tool:
    type: str
    shape: str
    handle: tuple
    head: tuple | None = None
    head_cyl: tuple | None = None
    bulb: tuple | None = None
    scale_norm: tuple = ()
    mass: float = 0.0
    ipos: tuple = (0.0, 0.0, 0.0)
    inertia: tuple = (0.0, 0.0, 0.0)
    head_pos_x: float = 0.0
    head_cyl_pos_x: float = 0.0
    bulb_pos_x: float = 0.0
    mesh: str | None = None
    mesh_parts: tuple = ()
    proxies: tuple = ()
    density: float = 0.0


def _read_obj_verts(path: Path) -> np.ndarray:
    verts = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                verts.append([float(value) for value in line.split()[1:4]])
    return np.asarray(verts, float)


def _read_obj(path: Path):
    verts, faces = [], []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                verts.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    return verts, faces


def _write_obj(path: Path, verts, faces, note: str) -> None:
    with open(path, "w") as handle:
        handle.write(f"# {note}\n")
        handle.writelines(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n" for point in verts)
        handle.writelines(f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}\n" for triangle in faces)


def _split_obj_groups(path: Path):
    groups: dict = {}
    current = None
    vcount = fcount = 0
    with open(path) as handle:
        lines = handle.readlines()
    for line in lines:
        if line.startswith(("o ", "g ")):
            current = line.split()[1]
            groups.setdefault(current, (set(), set()))
        elif line.startswith("v "):
            vcount += 1
            if current:
                groups[current][0].add(vcount - 1)
        elif line.startswith("f "):
            fcount += 1
            if current:
                groups[current][1].add(fcount - 1)
    return groups


def _cuboid_mass_inertia(scale, density: float):
    lx, ly, lz = scale
    mass = lx * ly * lz * density
    return (
        mass,
        (1 / 12) * mass * (ly * ly + lz * lz),
        (1 / 12) * mass * (lx * lx + lz * lz),
        (1 / 12) * mass * (lx * lx + ly * ly),
    )


def _capsule_mass_inertia(height: float, diameter: float, density: float):
    radius = diameter / 2
    mass_cylinder = density * math.pi * radius * radius * height
    mass_hemisphere = density * (2 / 3) * math.pi * radius**3
    mass = mass_cylinder + 2 * mass_hemisphere
    axis = 0.5 * mass_cylinder * radius * radius + 2 * (2 / 5) * mass_hemisphere * radius * radius
    com = (height / 2) + (3 * radius / 8)
    perpendicular = (1 / 12) * mass_cylinder * (3 * radius * radius + height * height) + 2 * (
        (83 / 320) * mass_hemisphere * radius * radius + mass_hemisphere * com * com
    )
    return mass, (perpendicular, perpendicular, axis)


def _part_mass_inertia(kind: str, scale, density: float):
    if kind == "box":
        mass, ixx, iyy, izz = _cuboid_mass_inertia(scale, density)
        return mass, (ixx, iyy, izz)
    mass, (ixx, iyy, izz) = _capsule_mass_inertia(scale[0], scale[1], density)
    return mass, (izz, iyy, ixx)


def _compose(parts):
    masses = [_part_mass_inertia(kind, scale, density)[0] for kind, scale, density, _ in parts]
    total = float(sum(masses))
    com = float(sum(mass * center for mass, (_, _, _, center) in zip(masses, parts)) / total)
    ixx = iyy = izz = 0.0
    for mass, (kind, scale, density, center) in zip(masses, parts):
        _, (a, b, c) = _part_mass_inertia(kind, scale, density)
        delta = center - com
        ixx += a
        iyy += b + mass * delta * delta
        izz += c + mass * delta * delta
    return total, com, (ixx, iyy, izz)


class ToolSampler:
    def __init__(self, config: Config, assets: AssetRegistry):
        self.config = config
        self.assets = assets
        self.distributions = tuple(ToolDistribution(**dict(item)) for item in config.distributions)
        self.types = tuple(config.types)
        self.default_types = tuple(config.default_types)

    def _ensure_pipette_mesh(self, verbose: bool = False):
        mesh = self.config.pipette.mesh
        body_part = self.assets.get(mesh.body_part)
        tube_part = self.assets.get(mesh.tube_part)
        body_part.path.parent.mkdir(parents = True, exist_ok = True)
        if body_part.path.exists() and tube_part.path.exists():
            return body_part.path

        raw = self.assets.get("pipette_raw").path
        try:
            if not raw.exists():
                import urllib.request

                urllib.request.urlretrieve(mesh.url, raw)
            verts, faces = _read_obj(raw)
            groups = _split_obj_groups(raw)
            tube_idx = groups.get("tube", (set(), set()))[0]
            points = np.asarray(verts, float) * float(mesh.unit_scale)
            points = points[:, [int(axis) for axis in mesh.axis_order]]
            points -= 0.5 * (points.max(0) + points.min(0))
            note = "AutoBio pipette (autobio-bench/AutoBio), metres, +x axis, shared bbox centre"
            body_v, tube_v, body_f, tube_f = [], [], [], []
            for triangle in faces:
                dst_v, dst_f = (tube_v, tube_f) if all(i in tube_idx for i in triangle) else (body_v, body_f)
                for i in triangle:
                    dst_v.append(points[i])
                dst_f.append([len(dst_v) - 3, len(dst_v) - 2, len(dst_v) - 1])
            _write_obj(body_part.path, body_v, body_f, note + " [body]")
            _write_obj(tube_part.path, tube_v, tube_f, note + " [tube]")
            if verbose:
                print(f"[pipette] cooked {len(faces)} faces -> {body_part.path.name}, {tube_part.path.name}")
            return body_part.path
        except Exception as error:  # noqa: BLE001 - offline / moved mesh -> procedural fallback
            if verbose:
                print(f"[pipette] mesh unavailable ({error}); using the procedural pipette")
            return None

    def _pipette_proxies(self, mesh_path: Path, scale: float = 1.0):
        mesh = self.config.pipette.mesh
        points = _read_obj_verts(mesh_path) * scale
        low, high = points[:, 0].min(), points[:, 0].max()
        radius = np.linalg.norm(points[:, 1:], axis = 1)
        min_half = float(self.config.min_half)
        proxies = []
        for low_fraction, high_fraction in mesh.proxy_spans:
            low_span = low + float(low_fraction) * (high - low)
            high_span = low + float(high_fraction) * (high - low)
            mask = (points[:, 0] >= low_span) & (points[:, 0] <= high_span)
            percentile = float(np.percentile(radius[mask], float(mesh.proxy_percentile))) if mask.any() else min_half
            proxies.append((max(min_half, percentile), 0.5 * (high_span - low_span), 0.5 * (low_span + high_span)))
        return tuple(proxies)

    def _build_pipette(self, rng, mesh_path: Path | None = None) -> Tool:
        config = self.config
        mesh_config = config.pipette.mesh
        object_base_size = float(config.object_base_size)
        min_half = float(config.min_half)
        density = float(rng.uniform(*config.pipette.ranges.density))

        if exists(mesh_path):
            mesh_scale = float(rng.uniform(*mesh_config.scale_range))
            proxies = self._pipette_proxies(mesh_path, mesh_scale)
            parts = [("cylinder", (2 * half_length, 2 * radius), density, x) for radius, half_length, x in proxies]
            mesh_parts = tuple(
                (f"{mesh_config.name}_{part.suffix}", tuple(part.rgba), mesh_scale)
                for part in mesh_config.parts
            )
            tool = Tool(
                type = "pipette",
                shape = "cylinder",
                handle = (2 * proxies[1][1], 2 * proxies[1][0]),
                mesh = mesh_config.name,
                proxies = proxies,
                mesh_parts = mesh_parts,
            )
            mass, com, inertia = _compose(parts)
            tool.mass, tool.ipos, tool.inertia = mass, (com, 0.0, 0.0), inertia
            lows = [x - half_length for _, half_length, x in proxies]
            highs = [x + half_length for _, half_length, x in proxies]
            length = max(highs) - min(lows)
            tool.scale_norm = tuple(float(v) / object_base_size for v in (length, 2 * proxies[1][0], 2 * proxies[1][0]))
        else:
            ranges = config.pipette.ranges
            bulb = (float(rng.uniform(*ranges.bulb_len)), float(rng.uniform(*ranges.bulb_dia)))
            barrel = (float(rng.uniform(*ranges.barrel_len)), float(rng.uniform(*ranges.barrel_dia)))
            tip = (float(rng.uniform(*ranges.tip_len)), float(rng.uniform(*ranges.tip_dia)))
            tool = Tool(type = "pipette", shape = "cylinder", handle = barrel, bulb = bulb, head_cyl = tip)
            tool.bulb_pos_x = -bulb[0] / 2
            tool.head_cyl_pos_x = barrel[0] + tip[0] / 2
            parts = [
                ("cylinder", bulb, density, tool.bulb_pos_x),
                ("cylinder", barrel, density, barrel[0] / 2),
                ("cylinder", tip, density, tool.head_cyl_pos_x),
            ]
            mass, com, inertia = _compose(parts)
            tool.mass, tool.ipos, tool.inertia = mass, (com, 0.0, 0.0), inertia
            tool.scale_norm = tuple(float(v) / object_base_size for v in (barrel[0], barrel[1], barrel[1]))
            tool.proxies = (
                (max(bulb[1] / 2, min_half), bulb[0] / 2, tool.bulb_pos_x),
                (max(barrel[1] / 2, min_half), barrel[0] / 2, barrel[0] / 2),
                (max(tip[1] / 2, min_half), tip[0] / 2, tool.head_cyl_pos_x),
            )
        tool.density = density
        return tool

    def _build_tool(
        self,
        handle_scale,
        head_scale,
        handle_density,
        head_density,
        type_: str = "hammer",
        shape: str = "box",
    ) -> Tool:
        parts = [(shape, handle_scale, handle_density, 0.0)]
        tool = Tool(type = type_, shape = shape, handle = tuple(float(value) for value in handle_scale))
        if exists(head_scale):
            head = tuple(float(value) for value in head_scale)
            tool.head_pos_x = handle_scale[0] / 2 + head[0] / 2
            parts.append(("box", head, head_density, tool.head_pos_x))
            tool.head = head
        mass, com, inertia = _compose(parts)
        tool.mass, tool.ipos, tool.inertia = mass, (com, 0.0, 0.0), inertia
        normalized = handle_scale if len(handle_scale) == 3 else (handle_scale[0], handle_scale[1], handle_scale[1])
        tool.scale_norm = tuple(float(value) / float(self.config.object_base_size) for value in normalized)
        return tool

    def sample(
        self,
        num_per_type: int | None = None,
        seed: int | None = None,
        types = None,
        shuffle: bool = True,
        verbose: bool = False,
    ) -> list[Tool]:
        num_per_type = int(default(num_per_type, self.config.pool.num_per_type))
        seed = int(default(seed, self.config.pool.seed))
        types = tuple(default(types, self.default_types))
        rng = np.random.RandomState(seed)
        tools: list[Tool] = []

        for distribution in (item for item in self.distributions if item.type in set(types)):
            handle_density = rng.uniform(*distribution.handle_density, num_per_type)
            head_density = (
                rng.uniform(*distribution.head_density, num_per_type)
                if exists(distribution.head_density)
                else None
            )
            handle_scale = rng.uniform(
                distribution.handle_min,
                distribution.handle_max,
                size = (num_per_type, len(distribution.handle_min)),
            )
            head_scale = (
                rng.uniform(
                    distribution.head_min,
                    distribution.head_max,
                    size = (num_per_type, len(distribution.head_min)),
                )
                if distribution.head_min
                else None
            )
            for index in range(num_per_type):
                tools.append(
                    self._build_tool(
                        tuple(handle_scale[index]),
                        tuple(head_scale[index]) if exists(head_scale) else None,
                        float(handle_density[index]),
                        float(head_density[index]) if exists(head_density) else None,
                        type_ = distribution.type,
                        shape = distribution.shape,
                    )
                )

        if "pipette" in set(types):
            mesh = self._ensure_pipette_mesh(verbose = verbose)
            tools += [self._build_pipette(rng, mesh) for _ in range(num_per_type)]
        if shuffle:
            tools = [tools[index] for index in rng.permutation(len(tools))]
        return tools


def build_tool_body(tool: Tool, config: Config, geometry: Config, object_z: float) -> str:
    tiny = float(config.tiny)
    min_half = float(config.min_half)
    cylinder_quat = geometry.cylinder_quat

    handle = tool.handle
    head = default(tool.head, (tiny, tiny, tiny))
    head_cyl = default(tool.head_cyl, (tiny, tiny))
    bulb = default(tool.bulb, (tiny, tiny))
    box_size = (handle[0] / 2, handle[1] / 2, handle[2] / 2) if tool.shape == "box" else (tiny, tiny, tiny)
    cyl_size = (tiny, tiny) if tool.shape == "box" else (handle[1] / 2, handle[0] / 2)

    visual = render(
        "play_tool_visual.xml",
        handle_box = vec(box_size),
        handle_cyl = vec(cyl_size),
        head_box = vec((head[0] / 2, head[1] / 2, head[2] / 2)),
        head_cyl = vec((head_cyl[1] / 2, head_cyl[0] / 2)),
        bulb_cyl = vec((bulb[1] / 2, bulb[0] / 2)),
        head_pos = f"{tool.head_pos_x:.6f}",
        head_cyl_pos = f"{tool.head_cyl_pos_x:.6f}",
        bulb_pos = f"{tool.bulb_pos_x:.6f}",
        handle_rgba = vec(config.colors.handle),
        head_rgba = vec(config.colors.head),
        bulb_rgba = vec(config.colors.bulb),
        cylinder_quat = cylinder_quat,
    )
    collision = render(
        "play_tool_collision.xml",
        handle_box = vec(tuple(max(value, min_half) for value in box_size)),
        handle_cyl = vec(tuple(max(value, min_half) for value in cyl_size)),
        head_box = vec(tuple(max(value, min_half) for value in (head[0] / 2, head[1] / 2, head[2] / 2))),
        head_cyl = vec(tuple(max(value, min_half) for value in (head_cyl[1] / 2, head_cyl[0] / 2))),
        bulb_cyl = vec(tuple(max(value, min_half) for value in (bulb[1] / 2, bulb[0] / 2))),
        head_pos = f"{tool.head_pos_x:.6f}",
        head_cyl_pos = f"{tool.head_cyl_pos_x:.6f}",
        bulb_pos = f"{tool.bulb_pos_x:.6f}",
        cylinder_quat = cylinder_quat,
    )

    mesh_parts = tool.mesh_parts or tuple(
        (f"{config.pipette.mesh.name}_{part.suffix}", tuple(part.rgba), tiny)
        for part in config.pipette.mesh.parts
    )
    meshes = "\n".join(
        render(
            "play_tool_mesh.xml",
            index = index,
            name = name,
            scale = vec((scale, scale, scale)),
            rgba = vec(rgba),
        )
        for index, (name, rgba, scale) in enumerate(mesh_parts)
    )
    proxies = (tool.proxies or ((min_half, min_half, 0.0),) * int(config.proxy_slots))[: int(config.proxy_slots)]
    proxy_xml = "\n".join(
        render(
            "play_tool_proxy.xml",
            index = index,
            radius = f"{radius:.6f}",
            half_length = f"{half_length:.6f}",
            pos = f"{center:.6f}",
            cylinder_quat = cylinder_quat,
        )
        for index, (radius, half_length, center) in enumerate(proxies)
    )

    return render(
        "play_object.xml",
        z = f"{object_z:.5f}",
        visual = visual,
        collision = collision,
        meshes = meshes,
        proxies = proxy_xml,
    )


def bake_tools(env, tools: list[Tool], config: Config) -> None:
    """Write per-world tool geometry, mass and inertia into the model arrays."""

    tiny = float(config.tiny)
    min_half = float(config.min_half)
    primitives = tuple(config.primitives)
    mesh_slots = int(config.mesh_slots)
    proxy_slots = int(config.proxy_slots)
    tints = config.colors.tints

    for world in range(env.nworld):
        tool = tools[world % len(tools)]
        tint = np.array(tints.get(tool.type, (1.0, 1.0, 1.0)))
        clear = np.zeros(4)

        for slot in primitives:
            env.geom_size[world, env._slot_geoms[f"{slot}_vis"]] = (tiny, tiny, tiny)
            env.geom_rgba[world, env._slot_geoms[f"{slot}_vis"]] = clear
            env.geom_size[world, env._slot_geoms[f"{slot}_phys"]] = (min_half, min_half, min_half)
            env.geom_pos[world, env._slot_geoms[f"{slot}_phys"]] = (0.0, 0.0, 0.0)

        for index in range(mesh_slots):
            mesh_id = env._slot_geoms.get(f"mesh{index}_vis")
            if not exists(mesh_id):
                continue
            scale = tool.mesh_parts[index][2] if (tool.mesh and index < len(tool.mesh_parts)) else tiny
            env.geom_size[world, mesh_id] = (scale, scale, scale)

        proxies = (tool.proxies or ((min_half, min_half, 0.0),) * proxy_slots)[:proxy_slots]
        for index, (radius, half_length, center) in enumerate(proxies):
            proxy_id = env._slot_geoms.get(f"proxy{index}_phys")
            if exists(proxy_id):
                env.geom_size[world, proxy_id] = (radius, half_length, min_half)
                env.geom_pos[world, proxy_id] = (center, 0.0, 0.0)

        if not exists(tool.mesh):
            handle_color = np.array(config.colors.handle)
            head_color = np.array(config.colors.head)
            bulb_color = np.array(config.colors.bulb)
            handle_rgba = np.concatenate([np.clip(handle_color[:3] * tint, 0, 1), handle_color[3:]])
            metal_rgba = np.concatenate([np.clip(head_color[:3] * tint, 0, 1), head_color[3:]])
            bulb_rgba = np.clip(bulb_color * np.append(tint, 1.0), 0, 1)
            handle = tool.handle
            if tool.shape == "box":
                env.geom_size[world, env._slot_geoms["handle_box_vis"]] = (handle[0] / 2, handle[1] / 2, handle[2] / 2)
                env.geom_rgba[world, env._slot_geoms["handle_box_vis"]] = handle_rgba
                env.geom_size[world, env._slot_geoms["handle_box_phys"]] = tuple(
                    max(value / 2, min_half) for value in handle
                )
            else:
                env.geom_size[world, env._slot_geoms["handle_cyl_vis"]] = (handle[1] / 2, handle[0] / 2, tiny)
                env.geom_rgba[world, env._slot_geoms["handle_cyl_vis"]] = handle_rgba
                env.geom_size[world, env._slot_geoms["handle_cyl_phys"]] = (
                    max(handle[1] / 2, min_half),
                    max(handle[0] / 2, min_half),
                    min_half,
                )
            if exists(tool.head):
                env.geom_size[world, env._slot_geoms["head_box_vis"]] = np.array(tool.head) / 2
                env.geom_pos[world, env._slot_geoms["head_box_vis"]] = (tool.head_pos_x, 0.0, 0.0)
                env.geom_rgba[world, env._slot_geoms["head_box_vis"]] = metal_rgba
                env.geom_size[world, env._slot_geoms["head_box_phys"]] = np.maximum(np.array(tool.head) / 2, min_half)
                env.geom_pos[world, env._slot_geoms["head_box_phys"]] = (tool.head_pos_x, 0.0, 0.0)
            if exists(tool.head_cyl):
                env.geom_size[world, env._slot_geoms["head_cyl_vis"]] = (
                    tool.head_cyl[1] / 2,
                    tool.head_cyl[0] / 2,
                    tiny,
                )
                env.geom_pos[world, env._slot_geoms["head_cyl_vis"]] = (tool.head_cyl_pos_x, 0.0, 0.0)
                env.geom_rgba[world, env._slot_geoms["head_cyl_vis"]] = metal_rgba
                env.geom_size[world, env._slot_geoms["head_cyl_phys"]] = (
                    max(tool.head_cyl[1] / 2, min_half),
                    max(tool.head_cyl[0] / 2, min_half),
                    min_half,
                )
                env.geom_pos[world, env._slot_geoms["head_cyl_phys"]] = (tool.head_cyl_pos_x, 0.0, 0.0)
            if exists(tool.bulb):
                env.geom_size[world, env._slot_geoms["bulb_cyl_vis"]] = (tool.bulb[1] / 2, tool.bulb[0] / 2, tiny)
                env.geom_pos[world, env._slot_geoms["bulb_cyl_vis"]] = (tool.bulb_pos_x, 0.0, 0.0)
                env.geom_rgba[world, env._slot_geoms["bulb_cyl_vis"]] = bulb_rgba
                env.geom_size[world, env._slot_geoms["bulb_cyl_phys"]] = (
                    max(tool.bulb[1] / 2, min_half),
                    max(tool.bulb[0] / 2, min_half),
                    min_half,
                )
                env.geom_pos[world, env._slot_geoms["bulb_cyl_phys"]] = (tool.bulb_pos_x, 0.0, 0.0)

        env.body_mass[world, env._obj_bid] = tool.mass
        env.body_inertia[world, env._obj_bid] = tool.inertia
        env.body_ipos[world, env._obj_bid] = tool.ipos
        env.body_iquat[world, env._obj_bid] = (1.0, 0.0, 0.0, 0.0)


# rewards

class RewardTerm:
    name = "term"

    def __init__(self, weight: float = 1.0, **params):
        self.weight = float(weight)
        self.params = params
        self._last: dict[str, np.ndarray] = {}

    def reset(self, env) -> None:
        self._last = {}

    def value(self, env) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return dict(self._last)

    def __call__(self, env) -> np.ndarray:
        value = np.asarray(self.value(env), float)
        self._last = {key: self.weight * np.asarray(item, float) for key, item in self._last.items()}
        return self.weight * value


class RewardComposer:
    def __init__(self, terms: list[RewardTerm]):
        self.terms = list(terms)

    @classmethod
    def from_config(cls, config: Config) -> RewardComposer:
        terms = []
        for spec in config.terms:
            params = {key: spec[key] for key in spec if key not in ("type", "weight")}
            terms.append(REWARDS.create(spec.type, weight = spec.get("weight", 1.0), **params))
        return cls(terms)

    def reset(self, env) -> None:
        for term in self.terms:
            term.reset(env)

    def compute(self, env) -> tuple[np.ndarray, dict]:
        reward = np.zeros(env.nworld, float)
        info = {}
        for term in self.terms:
            reward = reward + term(env)
            info.update(term.info())
        return reward, info


@REWARDS.register("lift")
class LiftReward(RewardTerm):
    name = "lift"

    def value(self, env):
        task = env.task
        progress_scale = float(self.params["progress_scale"])
        bonus = float(self.params["bonus"])
        threshold = float(self.params["threshold"])
        base = float(self.params["base"])
        low, high = (float(value) for value in self.params["clip"])

        lift = base + task.object_pos[:, 2] - task.object_init_z
        progress = np.clip(lift, low, high)
        lifted = (lift > threshold) | task.lifted
        bonus_value = bonus * (lifted & ~task.lifted)
        progress = progress * (~lifted) * progress_scale
        task.lifted = lifted
        self._last = {"lift": progress, "lift_bonus": bonus_value}
        return progress + bonus_value

    def info(self):
        return dict(self._last)


@REWARDS.register("fingertip_progress")
class FingertipProgressReward(RewardTerm):
    name = "fingertip_delta"

    def value(self, env):
        task = env.task
        scale = float(self.params["scale"])
        low, high = (float(value) for value in self.params["clip"])
        delta = np.clip(task._closest_ft - task.fingertip_dist, low, high).sum(-1) * (~task.lifted)
        task._closest_ft = np.minimum(task._closest_ft, task.fingertip_dist)
        value = delta * scale
        self._last = {self.name: value}
        return value


@REWARDS.register("keypoint_progress")
class KeypointProgressReward(RewardTerm):
    name = "keypoint"

    def value(self, env):
        task = env.task
        scale = float(self.params["scale"])
        low, high = (float(value) for value in self.params["clip"])
        delta = np.clip(task._closest_kp - task.keypoint_dist, low, high) * task.lifted
        task._closest_kp = np.minimum(task._closest_kp, task.keypoint_dist)
        value = delta * scale
        self._last = {self.name: value}
        return value


@REWARDS.register("joint_velocity_penalty")
class JointVelocityPenalty(RewardTerm):
    name = "penalty"

    def value(self, env):
        arm = float(self.params["arm"])
        hand = float(self.params["hand"])
        qvel = env.d.qvel.numpy()
        arm_index = env.task.arm_joint_count
        value = -(arm * np.abs(qvel[:, :arm_index]).sum(-1) + hand * np.abs(qvel[:, arm_index:]).sum(-1))
        self._last = {self.name: value}
        return value


@REWARDS.register("goal_bonus")
class GoalBonus(RewardTerm):
    name = "bonus"

    def value(self, env):
        value = float(self.params["value"])
        dwell_steps = float(self.params["dwell_steps"])
        reward = env.task.near * (value / dwell_steps)
        self._last = {self.name: reward}
        return reward


@REWARDS.register("distance")
class DistanceReward(RewardTerm):
    name = "distance"

    def value(self, env):
        reward = -env.task.distance
        self._last = {self.name: reward}
        return reward


@REWARDS.register("lift_height")
class LiftHeightReward(RewardTerm):
    name = "lift_height"

    def value(self, env):
        low, high = (float(value) for value in self.params["clip"])
        reward = np.clip(env.task.lift, low, high)
        self._last = {self.name: reward}
        return reward


@REWARDS.register("success_bonus")
class SuccessBonus(RewardTerm):
    name = "success_bonus"

    def value(self, env):
        reward = env.task.success.astype(float)
        self._last = {self.name: reward}
        return reward


# goals

@dataclass
class DwellSuccess:
    tolerance: float
    dwell_steps: int
    dwell: np.ndarray | None = field(default = None, init = False)

    def reset(self, count: int) -> None:
        self.dwell = np.zeros(count, dtype = int)

    def update(self, distance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not exists(self.dwell):
            self.reset(np.shape(distance)[0])
        near = np.asarray(distance) <= self.tolerance
        self.dwell = np.where(near, self.dwell + 1, 0)
        return near, self.dwell >= self.dwell_steps, self.dwell.copy()


class KeypointGoal:
    """Sample a goal pose and hold a tool-keypoint tolerance for a dwell."""

    def __init__(self, config: Config, dwell: DwellSuccess, scale: float):
        self.volume_min = np.array(config.volume_min, float)
        self.volume_max = np.array(config.volume_max, float)
        self.scale = float(scale)
        self.dwell = dwell
        self.goal_pose: np.ndarray | None = None

    def bind(self, env, task) -> None:
        self.goal_pose = np.zeros((env.nworld, 7))

    def reset(self, env, worlds: np.ndarray) -> None:
        count = len(worlds)
        rng = env.rng
        position = rng.uniform(self.volume_min, self.volume_max, (count, 3))
        quat = rand_quat_wxyz(count, rng)
        env.body_pos[worlds, env.task.goal_bid] = position
        env.body_quat[worlds, env.task.goal_bid] = quat
        self.goal_pose[worlds] = np.concatenate([position, quat], axis = 1)
        self.dwell.dwell[worlds] = 0

    def update(self, env) -> None:
        near, success, dwell = self.dwell.update(env.task.keypoint_dist / self.scale)
        env.task.near = near
        env.task.success = success
        env.task.dwell = dwell


SCENARIOS.register("keypoint", KeypointGoal)


# task protocol

class Task:
    """A task owns a scene, goal scenario, rewards and reset logic."""

    config: Config
    scene: Scene
    action_space: Any
    targets: np.ndarray

    def bind(self, env) -> None:
        """Cache model ids and allocate per-task state. Called once."""

    def patch_model(self, env) -> None:
        """Apply task-specific friction, damping and collision groups."""

    def reset(self, env, worlds: np.ndarray) -> None:
        raise NotImplementedError

    def apply_action(self, env, action: np.ndarray) -> None:
        raise NotImplementedError

    def update(self, env) -> None:
        raise NotImplementedError

    def observe(self, env) -> dict:
        raise NotImplementedError

    def random_action(self, env, mode: str = "smooth") -> np.ndarray:
        raise NotImplementedError

    def metrics(self, env) -> dict:
        return {}


# vectorized MuJoCo-Warp environment

@wp.kernel(enable_backward = False)
def _clamp_qvel(qvel: wp.array2d[wp.float32], limit: float):
    world, index = wp.tid()
    qvel[world, index] = wp.clamp(qvel[world, index], -limit, limit)


def _batched_model_fields() -> tuple[str, ...]:
    return tuple(
        field.name for field in dataclasses.fields(mjw.Model) if getattr(field.type, "shape", ())[:1] == ("*",)
    )


class MjWarpEnv:
    """Vectorized MuJoCo-Warp environment driven by a :class:`Task`."""

    def __init__(
        self,
        task,
        *,
        nworld: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        res: tuple | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        verbose: bool | None = None,
        xml: str | None = None,
        config: Config | None = None,
    ):
        self.config = default(config, task.config)
        defaults = self.config.environment.defaults
        self.task = task
        self.nworld = int(default(nworld, defaults.nworld))
        self.seed = int(default(seed, defaults.seed))
        self.res = tuple(default(res, defaults.resolution))
        self.verbose = bool(default(verbose, defaults.verbose))
        self.rng = np.random.RandomState(self.seed)
        self.simulation = self.config.simulation

        self.decimation = round(float(self.simulation.physics_hz) / float(self.simulation.policy_hz))
        self.qvel_clamp = float(self.simulation.qvel_clamp)

        if exists(device):
            wp.set_device(device)

        self.xml = default(xml, task.scene.xml)
        self.mjm = mujoco.MjModel.from_xml_string(self.xml)
        self._patch_host_model()
        self._batched = _batched_model_fields()
        self.m = mjw.put_model(self.mjm, batch_sizes = {name: self.nworld for name in self._batched})

        data = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, data)
        self.nconmax = int(default(nconmax, defaults.nconmax))
        self.njmax = int(default(njmax, defaults.njmax))
        self.d = mjw.put_data(
            self.mjm,
            data,
            nworld = self.nworld,
            nconmax = self.nconmax,
            njmax = self.njmax,
            nvmax = self.mjm.nv,
        )
        mjw.reset_data(self.m, self.d)
        if hasattr(self.m.opt, "warn_overflow"):
            self.m.opt.warn_overflow = 0

        self._alloc_world_arrays()
        self.task.bind(self)
        self._upload_world_arrays()
        self.reset()
        self.rc = None
        self._render_groups = tuple(self.config.render.groups)
        self._num_nan_resets = 0

        if self.verbose:
            print(
                f"[{self.config.name}] worlds={self.nworld} nq={self.mjm.nq} nv={self.mjm.nv} "
                f"ngeom={self.mjm.ngeom} nconmax={self.nconmax} device={wp.get_device()}"
            )

    # model
    def _patch_host_model(self) -> None:
        model = self.mjm
        for actuator in range(model.nu):
            joint = int(model.actuator_trnid[actuator, 0])
            model.actuator_ctrlrange[actuator] = model.jnt_range[joint]
            model.actuator_ctrllimited[actuator] = 1
        self.task.patch_model(self)

    def _alloc_world_arrays(self) -> None:
        model = self.mjm
        for name in (
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_rgba",
            "body_pos",
            "body_quat",
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
        ):
            value = getattr(model, name)
            tiled = (
                np.tile(value[None], (self.nworld, 1, 1))
                if value.ndim > 1
                else np.tile(value[None], (self.nworld, 1))
            )
            setattr(self, name, tiled)

    def _upload_world_arrays(self) -> None:
        for name in (
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_rgba",
            "body_pos",
            "body_quat",
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
        ):
            getattr(self.m, name).assign(getattr(self, name))

    # lifecycle
    def reset(self, worlds = None):
        worlds = np.atleast_1d(np.asarray(default(worlds, np.arange(self.nworld))))
        if hasattr(self.d, "qacc_warmstart"):
            warmstart = self.d.qacc_warmstart.numpy()
            warmstart[worlds] = 0.0
            self.d.qacc_warmstart = wp.array(warmstart, dtype = wp.float32)
        self.task.reset(self, worlds)
        return self.observe()

    def step(self, action):
        action = np.asarray(action, float)
        if action.ndim == 1:
            action = action[None, :]
        action = np.clip(action, -1.0, 1.0)
        self.task.apply_action(self, action)
        self.d.ctrl = wp.array(self.task.targets, dtype = wp.float32)
        for _ in range(self.decimation):
            mjw.step(self.m, self.d)
            wp.launch(_clamp_qvel, dim = (self.nworld, self.mjm.nv), inputs = [self.d.qvel, self.qvel_clamp])
        self.task.update(self)

        reward = self.reward.copy()
        done = self.done.copy()
        success = default(getattr(self, "success", None), np.zeros(self.nworld, bool)).copy()
        terminated = default(getattr(self, "terminated", None), np.zeros(self.nworld, bool)).copy()
        truncated = default(getattr(self, "truncated", None), np.zeros(self.nworld, bool)).copy()
        info = self.info

        finite = np.isfinite(self.d.qpos.numpy()).all(axis = 1)
        if not finite.all():
            bad = np.flatnonzero(~finite)
            self._num_nan_resets += len(bad)
            self.reset(bad)
        elif done.any():
            self.reset(np.flatnonzero(done))

        # keep the terminal step visible to callers after the auto-reset
        self.done = done
        self.success = success
        self.info = info
        return self.observe(), reward, terminated, truncated, info

    # interface
    def observe(self) -> dict:
        return self.task.observe(self)

    def random_action(self, mode: str = "smooth"):
        return self.task.random_action(self, mode)

    def render(self, res: tuple | None = None, groups: tuple | None = None) -> np.ndarray:
        res = tuple(default(res, self.res))
        groups = tuple(default(groups, self.config.render.groups))
        if not exists(self.rc) or tuple(self.rc.cam_res.numpy()[0]) != res or groups != self._render_groups:
            self._render_groups = groups
            self.rc = mjw.create_render_context(
                self.mjm,
                nworld = self.nworld,
                cam_res = [res],
                render_rgb = True,
                render_depth = False,
                render_skybox = bool(self.config.render.skybox),
                use_shadows = bool(self.config.render.shadows),
                shadow_light_fraction = float(self.config.render.shadow_light_fraction),
                background_color = tuple(self.config.render.background),
                enabled_geom_groups = list(groups),
            )
        mjw.refit_bvh(self.m, self.d, self.rc)
        mjw.render(self.m, self.d, self.rc)
        out = wp.zeros((self.nworld, res[1], res[0]), dtype = wp.vec3)
        mjw.get_rgb(self.rc, 0, out)
        return out.numpy()

    def close(self) -> None:
        self.rc = None

    def metrics(self) -> dict:
        return {
            "task": self.config.name,
            "nworld": self.nworld,
            "numerical_resets": int(self._num_nan_resets),
            **self.task.metrics(self),
        }


# play scene

def _camera(config: Config, camera = None):
    position, target = default(camera, (config.scene.camera.pos, config.scene.camera.lookat))
    return position, target, lookat_quat(position, target)


def play_goal_xml(config: Config) -> str:
    keypoint = config.tools.keypoint
    fixed = np.array(config.tools.fixed_size, float)
    half = 0.5 * float(keypoint.scale) * fixed
    corners = np.array(keypoint.corners, float)
    radius = float(keypoint.sphere_radius)
    material = keypoint.material

    parts = [
        render(
            "play_keypoint.xml",
            index = index,
            radius = f"{radius:.6f}",
            pos = vec(corner * half, precision = 5),
            material = material,
        )
        for index, corner in enumerate(corners)
    ]

    edges = (
        ((0, 1), 0, half[0]),
        ((2, 3), 0, half[0]),
        ((0, 2), 1, half[1]),
        ((1, 3), 1, half[1]),
        ((0, 3), 2, half[2]),
        ((1, 2), 2, half[2]),
    )
    for index, ((start, end), axis, length) in enumerate(edges):
        midpoint = 0.5 * (corners[start] + corners[end]) * half
        if axis == 2:
            quat = config.geometry.identity_quat
        elif axis == 0:
            quat = config.geometry.cylinder_quat
        else:
            quat = config.geometry.cylinder_quat_y
        parts.append(
            render(
                "play_keypoint_edge.xml",
                index = index,
                radius = f"{float(keypoint.edge_radius):.6f}",
                half_length = f"{length - radius / 2:.6f}",
                pos = vec(midpoint, precision = 5),
                quat = quat,
                material = material,
            )
        )

    return render(
        "play_goal.xml",
        z = f"{float(config.scene.table.z) + float(config.scene.table.object_z_offset):.5f}",
        parts = "\n".join(parts),
    )


def actuator_xml(config: Config) -> str:
    arm_stiffness = dict(zip(config.robot.arm.joints, config.robot.arm.stiffness))
    arm_damping = dict(zip(config.robot.arm.joints, config.robot.arm.damping))
    hand_stiffness = dict(zip(config.robot.hand.joints, config.robot.hand.stiffness))
    hand_damping = dict(zip(config.robot.hand.joints, config.robot.hand.damping))
    return "\n".join(
        render(
            "play_actuator.xml",
            name = joint,
            joint = config.robot.prefix + joint,
            kp = arm_stiffness.get(joint, hand_stiffness.get(joint)),
            kv = arm_damping.get(joint, hand_damping.get(joint)),
        )
        for joint in (*config.robot.arm.joints, *config.robot.hand.joints)
    )


def _mesh_assets(config: Config, assets: AssetRegistry, mesh_path = None) -> str:
    name = config.tools.pipette.mesh.name
    if exists(mesh_path):
        base = Path(mesh_path)
        paths = [
            (f"{name}_body", base.with_name(f"{name}_body.obj")),
            (f"{name}_tube", base.with_name(f"{name}_tube.obj")),
        ]
    else:
        paths = [
            (f"{name}_body", assets.path(config.tools.pipette.mesh.body_part)),
            (f"{name}_tube", assets.path(config.tools.pipette.mesh.tube_part)),
        ]
    return "\n".join(
        render("play_mesh_asset.xml", name = mesh_name, file = path.resolve())
        for mesh_name, path in paths
        if path.exists()
    )


def build_play_scene(
    config: Config,
    assets: AssetRegistry,
    tool: Tool,
    camera = None,
    mesh_path = None,
    goal_xml: str | None = None,
    extra_entities: tuple[Entity, ...] = (),
    scene: Scene | None = None,
) -> Scene:
    scene = default(scene, lambda: Scene(config.scene.model, assets, config.simulation))
    scene.add_asset_xml(render(
        "play_assets.xml",
        ghost_near = vec(config.tools.colors.ghosts.near),
        ghost_far = vec(config.tools.colors.ghosts.far),
    ))
    robot_bodies = scene.add_urdf("robot")
    scene.add_asset_xml(_mesh_assets(config, assets, mesh_path))

    scene.add_entity("lights", render("lights.xml"))
    scene.add_entity("floor", render("floor.xml"))
    position, _, quat = _camera(config, camera)
    scene.add_entity(
        "camera",
        render(
            "camera.xml",
            name = "view",
            pos = vec(position),
            quat = vec(quat),
            fovy = f"{float(config.scene.camera.fovy):.6f}",
        ),
    )
    table = config.scene.table
    scene.add_entity(
        "table",
        render(
            "play_table.xml",
            z = f"{float(table.z):.6f}",
            half_size = vec(tuple(float(value) / 2 for value in table.size)),
            rgba = vec(table.rgba),
            friction = vec(table.friction),
        ),
    )
    scene.add_entity(
        "robot",
        render("play_robot.xml", base_pos = vec(config.robot.base_pos), bodies = robot_bodies),
    )
    object_z = float(table.z) + float(table.object_z_offset)
    scene.add_entity("object", build_tool_body(tool, config.tools, config.geometry, object_z))
    for entity in extra_entities:
        scene.add_entity(entity.name, entity.xml)
    scene.add_entity("goal", default(goal_xml, lambda: play_goal_xml(config)))
    scene.add_actuator_xml(actuator_xml(config))
    return scene


def sample_tool_pool(
    num_per_type = None,
    seed = None,
    types = None,
    shuffle = True,
    config: Config | None = None,
) -> list[Tool]:
    config = default(config, PLAY_CONFIG)
    assets = AssetRegistry.from_config(config.assets)
    return ToolSampler(config.tools, assets).sample(
        num_per_type = num_per_type,
        seed = seed,
        types = default(types, config.tools.types),
        shuffle = shuffle,
    )


def build_scene_xml(
    task: str = "play",
    tools = None,
    camera = None,
    mesh_path = None,
    config: Config | None = None,
    entities: tuple[Entity, ...] = (),
    goal_xml: str | None = None,
) -> str:
    if task != "play":
        raise ValueError("build_scene_xml only builds the play scene")
    config = default(config, PLAY_CONFIG)
    assets = AssetRegistry.from_config(config.assets)
    sampler = ToolSampler(config.tools, assets)
    tool = default(tools, lambda: sampler.sample(types = DEFAULT_TOOL_TYPES))[0]
    scene = build_play_scene(
        config,
        assets,
        tool,
        camera = camera,
        mesh_path = mesh_path,
        goal_xml = goal_xml,
        extra_entities = entities,
    )
    return scene.xml()


# play task and environment

class PlayTask(Task):
    """Sample tools, hold tool-keypoint goals, and score play rewards."""

    def __init__(
        self,
        config: Config | None = None,
        assets: AssetRegistry | None = None,
        tools: list[Tool] | None = None,
        tool_types = None,
        camera = None,
        start_mode: str | None = None,
        scene: Scene | None = None,
    ):
        self.config = default(config, PLAY_CONFIG)
        self.arm_joints = tuple(self.config.robot.arm.joints)
        self.hand_joints = tuple(self.config.robot.hand.joints)
        self.joint_names = self.arm_joints + self.hand_joints
        self.arm_joint_count = len(self.arm_joints)
        self.arm_default = dict(zip(self.arm_joints, self.config.robot.arm.default_q))
        self.assets = default(assets, lambda: AssetRegistry.from_config(self.config.assets))
        self.sampler = ToolSampler(self.config.tools, self.assets)
        tool_types = default(tool_types, self.config.tools.default_types)
        self.tools = list(default(tools, lambda: self.sampler.sample(types = tool_types)))
        if not self.tools:
            raise ValueError("at least one tool is required")
        self.tool = self.tools[0]
        self.start_mode = default(start_mode, self.config.environment.defaults.start_mode)
        if self.start_mode not in tuple(self.config.environment.start_modes):
            raise ValueError(f"unknown start_mode: {self.start_mode}")
        self.scene = default(scene, lambda: build_play_scene(self.config, self.assets, self.tool, camera = camera))
        self.action_space = InferredSpace((len(self.joint_names),), -1.0, 1.0)

        success = self.config.task.success
        self.success_criteria = DwellSuccess(float(success.tolerance), int(success.dwell_steps))
        self.goal = SCENARIOS.create(
            self.config.scene.goal.get("type", "keypoint"),
            self.config.scene.goal,
            self.success_criteria,
            float(self.config.tools.keypoint.scale),
        )
        self.reward = RewardComposer.from_config(self.config.task.reward)
        self.near = np.zeros(1, bool)
        self.success = np.zeros(1, bool)
        self.dwell = np.zeros(1, int)

    # binding
    def bind(self, env) -> None:
        model = env.mjm

        def body_id(name):
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            return index if index >= 0 else None

        def geom_id(name):
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            return index if index >= 0 else None

        self.env = env
        self.arm_qadr = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.config.robot.prefix}{joint}")]
            for joint in self.arm_joints
        ])
        self.hand_qadr = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.config.robot.prefix}{joint}")]
            for joint in self.hand_joints
        ])
        self.qadr = np.concatenate([self.arm_qadr, self.hand_qadr])
        joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.config.robot.prefix}{joint}")
            for joint in self.joint_names
        ]
        self.joint_lo = np.array([model.jnt_range[joint][0] for joint in joints])
        self.joint_hi = np.array([model.jnt_range[joint][1] for joint in joints])
        self.default_joint_pos = np.array([self.arm_default.get(joint, 0.0) for joint in self.joint_names])

        self.object_bid = body_id("object")
        self.goal_bid = body_id("goal")
        self.palm_bid = body_id(self.config.robot.palm.body)
        self.fingertip_bids = np.array([body_id(name) for name in self.config.robot.fingertips.bodies])
        self.object_free = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
        self.object_dofadr = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
        self.ghost_geoms = np.array([
            geom for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(("ghost_kp_", "ghost_edge_"))
        ])
        self.kp_offsets = np.array(self.config.tools.keypoint.corners) * (
            0.5 * float(self.config.tools.keypoint.scale) * np.array(self.config.tools.fixed_size)
        )
        self.slot_geoms = {
            f"{base}{suffix}": geom_id(f"{base}{suffix}")
            for base in self.config.tools.primitives
            for suffix in ("_vis", "_phys")
        }
        for index in range(int(self.config.tools.mesh_slots)):
            self.slot_geoms[f"mesh{index}_vis"] = geom_id(f"mesh{index}_vis")
        for index in range(int(self.config.tools.proxy_slots)):
            self.slot_geoms[f"proxy{index}_phys"] = geom_id(f"proxy{index}_phys")

        env._obj_bid = self.object_bid
        env._goal_bid = self.goal_bid
        env._slot_geoms = self.slot_geoms
        env._ghost_geoms = self.ghost_geoms

        self.targets = np.tile(self.default_joint_pos, (env.nworld, 1))
        self.prev_targets = self.targets.copy()
        self.ghost_near = np.zeros(env.nworld, bool)
        self.successes = np.zeros(env.nworld, int)
        self.episode_len = np.zeros(env.nworld, int)
        self.closest_kp = np.full(env.nworld, -1.0)
        self.closest_ft = np.full((env.nworld, len(self.config.robot.fingertips.bodies)), -1.0)
        self.lifted = np.zeros(env.nworld, bool)
        self.object_init_z = np.zeros(env.nworld)
        self.noise = np.zeros((env.nworld, len(self.joint_names)))
        self.done = np.zeros(env.nworld, bool)
        self.reward_value = np.zeros(env.nworld)
        self.last_reward_terms: dict = {}
        self.cache: dict = {}

        self.success_criteria.reset(env.nworld)
        self.goal.bind(env, self)
        self.reward.reset(env)
        bake_tools(env, self.tools, self.config.tools)
        self._paint_ghosts(env)

    def patch_model(self, env) -> None:
        model = env.mjm
        robot_friction = self.config.robot.friction
        default_friction = float(robot_friction.default)
        fingertip_friction = float(robot_friction.fingertip)
        contact = tuple(robot_friction.contact)
        geom_name = {g: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") for g in range(model.ngeom)}
        body_name = {b: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in range(model.nbody)}

        for geom in range(model.ngeom):
            body, name = body_name[int(model.geom_bodyid[geom])], geom_name[geom]
            if body.endswith("_DP") or "elastomer" in name:
                friction = fingertip_friction
            elif body.startswith("r_") or name.startswith("r_"):
                friction = default_friction
            else:
                friction = float(self.config.simulation.default_geom.friction[0])
            model.geom_friction[geom] = (friction, contact[0], contact[1])

        hand_damping = dict(zip(self.hand_joints, self.config.robot.hand.damping))
        hand_armature = dict(zip(self.hand_joints, self.config.robot.hand.armature))
        hand_friction = dict(zip(self.hand_joints, self.config.robot.hand.frictionloss))
        for joint in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.config.robot.prefix}{joint}")
            if joint_id < 0:
                continue
            address = int(model.jnt_dofadr[joint_id])
            model.dof_damping[address] = hand_damping.get(joint, model.dof_damping[address])
            model.dof_armature[address] = hand_armature.get(joint, model.dof_armature[address])
            model.dof_frictionloss[address] = hand_friction.get(joint, model.dof_frictionloss[address])

        groups = self.config.collision.groups
        masks = {
            int(groups.robot): int(groups.object) | int(groups.world),
            int(groups.object): int(groups.robot) | int(groups.world),
            int(groups.world): int(groups.robot) | int(groups.object),
        }
        for geom in range(model.ngeom):
            if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
                continue
            name, body = geom_name[geom], body_name[int(model.geom_bodyid[geom])]
            if name.startswith("ghost_"):
                group = 0
            elif body.startswith("r_") or body == "robot":
                group = int(groups.robot)
            elif body == "object" or body.startswith("o_"):
                group = int(groups.object)
            else:
                group = int(groups.world)
            model.geom_contype[geom] = group
            model.geom_conaffinity[geom] = masks.get(group, 0)

    # reset
    def reset(self, env, worlds: np.ndarray) -> None:
        rng = env.rng
        qpos = env.d.qpos.numpy()
        qvel = env.d.qvel.numpy()
        reset_config = self.config.task.reset

        sampled = self.joint_lo + (self.joint_hi - self.joint_lo) * rng.random((len(worlds), len(self.joint_names)))
        scale = np.where(
            [joint in self.arm_default for joint in self.joint_names],
            float(reset_config.arm_dof_noise),
            float(reset_config.finger_dof_noise),
        )
        joint_pos = np.clip(
            self.default_joint_pos + scale * (sampled - self.default_joint_pos),
            self.joint_lo,
            self.joint_hi,
        )
        qpos[np.ix_(worlds, self.qadr)] = joint_pos
        qvel[np.ix_(worlds, self.qadr)] = rng.uniform(
            -float(reset_config.velocity_noise),
            float(reset_config.velocity_noise),
            size = (len(worlds), len(self.joint_names)),
        )
        self.targets[worlds] = joint_pos
        self.prev_targets[worlds] = joint_pos

        position_noise = reset_config.position_noise
        object_xy = np.stack(
            [
                rng.uniform(-float(position_noise[0]), float(position_noise[0]), len(worlds)),
                rng.uniform(-float(position_noise[1]), float(position_noise[1]), len(worlds)),
            ],
            axis = 1,
        )
        table = self.config.scene.table
        if self.start_mode == "paper":
            object_z = (
                float(table.z)
                + float(table.object_z_offset)
                + rng.uniform(-float(position_noise[2]), float(position_noise[2]), len(worlds))
            )
            object_quat = rand_quat_wxyz(len(worlds), rng)
        else:
            half_height = self._object_rest_half_height(env, worlds)
            gap, jitter = (float(value) for value in reset_config.table_gap)
            object_z = (
                float(table.z)
                + float(table.size[2]) / 2
                + half_height
                + gap
                + rng.uniform(0.0, jitter, len(worlds))
            )
            object_quat = yaw_quat_wxyz(len(worlds), rng)

        qpos[np.ix_(worlds, np.arange(self.object_free, self.object_free + 3))] = np.concatenate(
            [object_xy, object_z[:, None]], axis = 1
        )
        qpos[np.ix_(worlds, np.arange(self.object_free + 3, self.object_free + 7))] = object_quat
        qvel[np.ix_(worlds, np.arange(self.object_dofadr, self.object_dofadr + 6))] = 0.0
        self.object_init_z[worlds] = object_z

        self.goal.reset(env, worlds)
        self.successes[worlds] = 0
        self.episode_len[worlds] = -1  # update() below advances to 0
        self.closest_kp[worlds] = -1.0
        self.closest_ft[worlds] = -1.0
        self.lifted[worlds] = False
        self.ghost_near[worlds] = False
        self.done[worlds] = False

        env.d.qpos = wp.array(qpos, dtype = wp.float32)
        env.d.qvel = wp.array(qvel, dtype = wp.float32)
        env.d.ctrl = wp.array(self.targets, dtype = wp.float32)
        self._paint_ghosts(env)
        env._upload_world_arrays()
        mjw.forward(env.m, env.d)
        self.update(env)

    def _descendant_bodies(self, env, root):
        bodies = [root]
        changed = True
        while changed:
            changed = False
            for body in range(env.mjm.nbody):
                if body not in bodies and env.mjm.body_parentid[body] in bodies:
                    bodies.append(body)
                    changed = True
        return bodies

    def _object_rest_half_height(self, env, worlds):
        bodies = self._descendant_bodies(env, self.object_bid)
        geoms = np.flatnonzero(np.isin(env.mjm.geom_bodyid, bodies) & (env.mjm.geom_contype != 0))
        size, position = env.geom_size[np.ix_(worlds, geoms)], env.geom_pos[np.ix_(worlds, geoms)]
        types, aabb = np.asarray(env.mjm.geom_type)[geoms], np.asarray(env.mjm.geom_aabb)[geoms]
        half_height = size[..., 2].copy()
        is_cylinder = types == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        half_height[:, is_cylinder] = size[:, is_cylinder, 0]
        is_mesh = types == int(mujoco.mjtGeom.mjGEOM_MESH)
        if is_mesh.any():
            half_height[:, is_mesh] = np.abs(position[:, is_mesh, 2] + aabb[is_mesh, 2]) + aabb[is_mesh, 5]
        return np.max(np.abs(position[..., 2]) + half_height, axis = 1)

    # control
    def apply_action(self, env, action: np.ndarray) -> None:
        action_config = self.config.action
        moving_average = float(action_config.moving_average)
        speed = float(action_config.dof_speed_scale)
        policy_hz = float(self.config.simulation.policy_hz)

        arm_raw = np.clip(
            self.prev_targets[:, : self.arm_joint_count]
            + speed * (1.0 / policy_hz) * action[:, : self.arm_joint_count],
            self.joint_lo[: self.arm_joint_count],
            self.joint_hi[: self.arm_joint_count],
        )
        arm = np.clip(
            moving_average * arm_raw + (1 - moving_average) * self.prev_targets[:, : self.arm_joint_count],
            self.joint_lo[: self.arm_joint_count],
            self.joint_hi[: self.arm_joint_count],
        )
        hand_raw = self.joint_lo[self.arm_joint_count:] + 0.5 * (action[:, self.arm_joint_count:] + 1.0) * (
            self.joint_hi[self.arm_joint_count:] - self.joint_lo[self.arm_joint_count:]
        )
        hand = np.clip(
            moving_average * hand_raw + (1 - moving_average) * self.prev_targets[:, self.arm_joint_count:],
            self.joint_lo[self.arm_joint_count:],
            self.joint_hi[self.arm_joint_count:],
        )
        self.targets = np.concatenate([arm, hand], axis = 1)

    def _update_kinematics(self, env) -> None:
        xpos, xquat = env.d.xpos.numpy(), env.d.xquat.numpy()
        qvel = env.d.qvel.numpy()
        object_pos, object_quat = xpos[:, self.object_bid], xquat[:, self.object_bid]
        goal_pose = self.goal.goal_pose
        goal_pos, goal_quat = goal_pose[:, :3], goal_pose[:, 3:]
        keypoints = self._keypoints_world(object_pos, object_quat, self.kp_offsets)
        goal_keypoints = self._keypoints_world(goal_pos, goal_quat, self.kp_offsets)
        self.keypoint_dist = np.linalg.norm(keypoints - goal_keypoints, axis = -1).max(axis = -1)
        fingertip_pos = xpos[:, self.fingertip_bids]
        self.fingertip_dist = np.linalg.norm(fingertip_pos - object_pos[:, None, :], axis = -1)
        sentinel = self.closest_kp < 0.0
        self.closest_kp = np.where(sentinel, self.keypoint_dist, self.closest_kp)
        self.closest_ft = np.where(self.closest_ft < 0.0, self.fingertip_dist, self.closest_ft)
        self._closest_kp, self._closest_ft = self.closest_kp, self.closest_ft
        self.object_pos, self.object_quat = object_pos, object_quat
        self.goal_pos, self.goal_quat = goal_pos, goal_quat
        self.distance = self.keypoint_dist
        self.cache = {
            "xpos": xpos,
            "xquat": xquat,
            "qvel": qvel,
            "obj_pos": object_pos,
            "obj_quat": object_quat,
            "kp": keypoints,
            "goal_kp": goal_keypoints,
            "qpos": env.d.qpos.numpy(),
        }

    @staticmethod
    def _keypoints_world(center, quat, offsets) -> np.ndarray:
        rotated = np.array([quat_rotate_wxyz(q, offset) for q in quat for offset in offsets])
        return center[:, None, :] + rotated.reshape(len(center), len(offsets), 3)

    def update(self, env) -> None:
        self.episode_len += 1
        self._update_kinematics(env)
        self.goal.update(env)
        self._paint_ghosts(env, self.near)
        self.reward_value, self.last_reward_terms = self.reward.compute(env)

        new_success = self.success & (self.dwell == int(self.config.task.success.dwell_steps))
        self.successes += new_success.astype(int)
        self.episode_len[new_success] = 0

        termination = self.config.task.termination
        fall = self.object_pos[:, 2] < float(termination.object_fall_z)
        hand_far = self.fingertip_dist.max(-1) > float(termination.hand_far)
        max_successes = self.successes >= int(self.config.task.success.max_consecutive)
        timeout = self.episode_len >= int(self.config.task.episode.length)
        self.done = fall | hand_far | max_successes | timeout
        env.done = self.done
        env.success = self.success
        env.reward = self.reward_value
        env.last_reward_terms = self.last_reward_terms
        env.info = {
            "success": self.near,
            "is_success": self.success,
            "dwell": self.dwell,
            "difficulty": self.keypoint_dist,
            "distance": self.keypoint_dist,
        }
        env.terminated = np.zeros(env.nworld, bool)
        env.truncated = np.zeros(env.nworld, bool)
        self.prev_targets = self.targets.copy()

    def _paint_ghosts(self, env, near = None) -> None:
        if exists(near):
            self.ghost_near = near
        if len(self.ghost_geoms) == 0:
            return
        colors = self.config.tools.colors.ghosts
        near_color = np.array(colors.near, float)
        far_color = np.array(colors.far, float)
        env.geom_rgba[:, self.ghost_geoms] = np.where(
            self.ghost_near[:, None, None],
            near_color[None, None, :],
            far_color[None, None, :],
        )

    # observations
    def observe(self, env) -> dict:
        cache = self.cache
        qpos = cache["qpos"]
        joint_pos = 2.0 * (qpos[:, self.qadr] - self.joint_lo) / (self.joint_hi - self.joint_lo + 1e-12) - 1.0
        palm_offset = np.array([
            quat_rotate_wxyz(q, self.config.robot.palm.offset)
            for q in cache["xquat"][:, self.palm_bid]
        ])
        fingertip_offset = np.array([
            [quat_rotate_wxyz(q, self.config.robot.fingertips.offset) for q in cache["xquat"][:, body]]
            for body in self.fingertip_bids
        ]).transpose(1, 0, 2)
        palm_pos = cache["xpos"][:, self.palm_bid] + palm_offset
        fingertip_pos = cache["xpos"][:, self.fingertip_bids] + fingertip_offset
        object_scales = np.array([
            self.tools[index % len(self.tools)].scale_norm for index in range(env.nworld)
        ])
        return {
            "joint_pos": joint_pos,
            "joint_vel": cache["qvel"][:, self.qadr],
            "prev_action_targets": self.prev_targets,
            "palm_pos": palm_pos,
            "palm_rot": cache["xquat"][:, self.palm_bid],
            "object_rot": cache["obj_quat"],
            "fingertip_pos_rel_palm": (fingertip_pos - palm_pos[:, None, :]).reshape(env.nworld, -1),
            "keypoints_rel_palm": (cache["kp"] - palm_pos[:, None, :]).reshape(env.nworld, -1),
            "keypoints_rel_goal": (cache["kp"] - cache["goal_kp"]).reshape(env.nworld, -1),
            "object_scales": object_scales,
        }

    def random_action(self, env, mode: str = "smooth") -> np.ndarray:
        if mode == "hold":
            hand = (
                2.0 * (self.prev_targets[:, self.arm_joint_count:] - self.joint_lo[self.arm_joint_count:])
                / (self.joint_hi[self.arm_joint_count:] - self.joint_lo[self.arm_joint_count:] + 1e-9)
                - 1.0
            )
            return np.clip(
                np.concatenate([np.zeros((env.nworld, self.arm_joint_count)), hand], axis = 1),
                -1.0,
                1.0,
            )
        if mode == "zero":
            return np.zeros((env.nworld, len(self.joint_names)))
        if mode == "uniform":
            return env.rng.uniform(-1.0, 1.0, (env.nworld, len(self.joint_names)))

        config = self.config.task.random_action
        self.noise = np.clip(
            float(config.smoothing) * self.noise + float(config.noise) * env.rng.uniform(
                -float(config.noise_scale), float(config.noise_scale), self.noise.shape
            ),
            -1.0,
            1.0,
        )
        qpos = env.d.qpos.numpy()[:, self.qadr]
        pull = float(config.pull) * (self.default_joint_pos - qpos) / (self.joint_hi - self.joint_lo + 1e-9)
        return np.clip(self.noise + pull, -1.0, 1.0)

    def metrics(self, env) -> dict:
        return {
            "keypoint_dist_mean": float(np.mean(self.keypoint_dist)),
            "success_rate": float(np.mean(self.successes > 0)),
            "successes_mean": float(np.mean(self.successes)),
            "contacts": int(env.d.nacon.numpy().sum()),
            "tolerance": float(self.success_criteria.tolerance),
            "tools": [self.tools[index % len(self.tools)].type for index in range(env.nworld)],
        }


class PlayEnv(MjWarpEnv):
    def __init__(
        self,
        task: str = "play",
        nworld: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        tool_types = None,
        start_mode: str | None = None,
        camera = None,
        res: tuple | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        verbose: bool | None = None,
        tools = None,
        xml: str | None = None,
        config: Config | None = None,
    ):
        if task != "play":
            raise ValueError("only the play task is available in this environment")
        config = default(config, PLAY_CONFIG)
        play_task = PlayTask(
            config,
            tools = tools,
            tool_types = tool_types,
            camera = camera,
            start_mode = start_mode,
        )
        super().__init__(
            play_task,
            nworld = nworld,
            seed = seed,
            device = device,
            res = res,
            nconmax = nconmax,
            njmax = njmax,
            verbose = verbose,
            xml = xml,
            config = config,
        )


def tile_grid(images, cols = None, pad: int | None = None) -> np.ndarray:
    config = PLAY_CONFIG.demo
    pad = int(default(pad, config.grid_pad))
    count, height, width, _ = images.shape
    cols = default(cols, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    grid = np.full(
        (rows * height + (rows - 1) * pad, cols * width + (cols - 1) * pad, 3),
        float(config.grid_background),
        np.float32,
    )
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        grid[
            row * (height + pad): row * (height + pad) + height,
            col * (width + pad): col * (width + pad) + width,
        ] = image
    return (np.clip(grid, 0, 1) * 255).astype(np.uint8)


def demo(
    task = "play",
    nworld = None,
    steps = None,
    out = "out/demo",
    tool_types = None,
    random = None,
):
    from PIL import Image

    config = PLAY_CONFIG
    nworld = int(default(nworld, config.demo.nworld))
    steps = int(default(steps, config.demo.steps))
    random = default(random, config.demo.random)
    if isinstance(tool_types, str):
        tool_types = tuple(tool_types.split(","))
    env = PlayEnv(
        task,
        nworld = nworld,
        tool_types = tool_types,
        res = tuple(config.demo.resolution),
        verbose = True,
    )
    for _ in range(steps):
        env.step(env.random_action(random))
    Image.fromarray(tile_grid(env.render(), math.ceil(math.sqrt(nworld)))).save(out + ".png")
    print(env.metrics())
    env.close()


@TASKS.register("play")
def _make_play(**kwargs):
    return PlayEnv(**kwargs)


# public facade

Play2PerfectEnv = PlayEnv
Env = PlayEnv


def make(name: str = "play", **kwargs):
    if name == "pick" and "pick" not in TASKS:
        import play2perfect_core  # noqa: F401 - registers the pick task

    return TASKS.create(name, **kwargs)


if __name__ == "__main__":
    from fire import Fire

    Fire({"demo": demo})
