"""Types utilities for ROAM-Bench."""
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union, runtime_checkable
InstanceId = str

@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw_deg: float
    horizon_deg: float = 0.0
    standing: bool = True

@dataclass
class Obs:
    rgb: Any
    depth: Optional[Any] = None
    pose: Optional[Pose] = None
    step_idx: int = 0

@dataclass
class ActionResult:
    success: bool
    reason: str
    moved_m: float = 0.0
    turned_deg: float = 0.0
    collided: bool = False
    keyframe_idx: Optional[int] = None
    requested_m: Optional[float] = None

@dataclass(frozen=True)
class NameRef:
    name: InstanceId

@dataclass(frozen=True)
class PixelRef:
    u: int
    v: int

@dataclass(frozen=True)
class DirectionRef:
    deg: float
    dist_m: float
Referent = Union[NameRef, PixelRef, DirectionRef]
HELD_UNNAMED = '(something)'

@dataclass(frozen=True)
class WorldView:
    rooms: tuple = ()
    furniture: tuple = ()
    objects: tuple = ()
    relations: tuple = ()
    holding: Optional[InstanceId] = None
    here: Optional[str] = None
    visible: tuple = ()
    blind: bool = False
    hand_only: bool = False

    def __post_init__(self):
        if self.blind and self.hand_only:
            raise ValueError('WorldView cannot be both blind and hand_only: the first says there is no text observation at all and the second says there is exactly one line of it. Pick the mode the episode actually ran.')

    def text(self) -> str:
        if self.blind:
            return ''
        if self.hand_only:
            return f"holding: {self.holding or '(nothing)'}"

        def line(label, xs):
            return f'{label}: ' + (', '.join(xs) if xs else '(none known yet)')
        nowhere = '(no room -- a doorway, or outside every room)'
        parts = ['you are standing in: ' + (self.here or nowhere), 'in view right now: ' + (', '.join(self.visible) if self.visible else '(nothing you can name)'), line('rooms in this house', self.rooms), line('furniture you know of', self.furniture), line('objects you have seen', self.objects)]
        if self.relations:
            parts.append('placement: ' + '; '.join((f'{a} {r} {b}' for a, r, b in self.relations)))
        parts.append(f"holding: {self.holding or '(nothing)'}")
        return '\n'.join(parts)

@dataclass(frozen=True)
class ProposeResult:
    accepted: bool
    granted: Optional[InstanceId]
    reason: str
    phantom: bool = False

@dataclass(frozen=True)
class InstanceFacts:
    alias: InstanceId
    kind: str
    room: Optional[str] = None
    room_id: Optional[str] = None
    on: tuple = ()
    pickupable: bool = False
    openable: bool = False
    toggleable: bool = False
    sliceable: bool = False
    breakable: bool = False
    receptacle: bool = False
    is_open: Optional[bool] = None
    is_toggled: Optional[bool] = None
    footprint: Optional[float] = None
    look: Optional[str] = None
    inside: tuple = ()

@dataclass
class Episode:
    episode_id: str
    scene: str
    instruction: str = ''
    agent_pose: Optional[Pose] = None
    object_poses: Optional[list] = None
    initial_states: tuple = ()
    gt: dict = field(default_factory=dict)
    partial_obs: bool = True
    give_depth: bool = False
    give_pose: bool = False
    view_min_pixel_fraction: Optional[float] = None
    strict_object_discovery: bool = True
    furniture_prior: bool = True
    room_prior: bool = True
    furniture_prior_kinds_only: bool = False
    propose_max_distance: float = 1.5
    propose_feedback: str = 'verdict'
    world_text: bool = True
    action_feedback: str = 'verdict'
    hand_only: bool = False
    mark_pixels: str = 'off'
    view_min_pixels: int = 16

@runtime_checkable
class SimBackend(Protocol):

    def reset(self, episode: Episode) -> Obs:
        ...

    def close(self) -> None:
        ...

    def observe(self) -> Obs:
        ...

    def world(self) -> WorldView:
        ...

    def propose(self, text: str) -> ProposeResult:
        ...

    def move(self, distance_m: float) -> ActionResult:
        ...

    def turn(self, degrees: float) -> ActionResult:
        ...

    def look(self, delta_pitch_deg: float) -> ActionResult:
        ...

    def navigate_to_relative(self, deg: float, dist_m: float) -> ActionResult:
        ...

    def go_to_visible_object(self, ref: Referent) -> ActionResult:
        ...

    def navigate_to_object(self, ref: Referent) -> ActionResult:
        ...

    def interact(self, verb: str, ref: Referent, into: Optional[Referent]=None) -> ActionResult:
        ...

    def visible_instances(self, min_pixel_frac: Optional[float]=None) -> set:
        ...

    def instance_at_pixel(self, u: int, v: int) -> Optional[InstanceId]:
        ...

    def instances_in_patch(self, u: int, v: int, k: int=2) -> set:
        ...

    def instance_centroid(self, alias: InstanceId) -> Optional[tuple]:
        ...

    def bearing_to(self, alias: InstanceId) -> Optional[tuple]:
        ...

    def predicate(self, expr: str) -> bool:
        ...

    def agent_pose(self) -> Pose:
        ...

    def mutate(self, op: str, alias: InstanceId, arg) -> bool:
        ...

    def inventory(self) -> tuple:
        ...

    def room_labels(self) -> tuple:
        ...

    def agent_room(self) -> Optional[str]:
        ...

    def agent_room_name(self) -> Optional[str]:
        ...
