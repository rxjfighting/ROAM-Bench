"""Tiers utilities for ROAM-Bench."""
from dataclasses import dataclass
from .types import PixelRef
MANIPULATION = ('pick', 'place', 'open', 'close')
REFERRING = ('name', 'pixel')
PIXEL_SWAP = {'go_to_visible_object': 'go_to_object_at_pixel', 'navigate_to_relative': 'go_to_ground_at_pixel'}

@dataclass(frozen=True)
class Tier:
    key: str
    label: str
    navigation: tuple
    gives_away: str
    owns: str

    @property
    def skills(self):
        return self.navigation + MANIPULATION
TIERS = {
    'high': Tier(key='high', label='Privileged oracle navigation',
                 navigation=('navigate_to_object', 'look'),
                 gives_away='Navigation to a named object without a visibility requirement.',
                 owns='Ordering the task interactions.'),
    'plan': Tier(key='plan', label='Search and object handling',
                 navigation=('go_to_visible_object', 'navigate_to_relative', 'move', 'strafe', 'turn', 'look'),
                 gives_away='Local path planning and approach to a visible target.',
                 owns='Selecting observations, navigation targets, and interactions.'),
    'spatial': Tier(key='spatial', label='Metric and reference-frame reasoning',
                    navigation=('move', 'strafe', 'turn', 'look'),
                    gives_away='Execution of specified translations and rotations.',
                    owns='Choosing movement distances, directions, and viewpoints.'),
}
ORDER = ('high', 'plan', 'spatial')

class TierError(AttributeError):
    pass

class ActionSpace:

    def __init__(self, sim, tier, refer='name'):
        self._sim = sim
        self.tier = TIERS[tier] if isinstance(tier, str) else tier
        if refer not in REFERRING:
            raise ValueError(f'refer must be one of {REFERRING}, not {refer!r}')
        if refer == 'pixel' and 'navigate_to_object' in self.tier.navigation:
            raise ValueError("the high tier cannot use the pixel channel: navigate_to_object reaches objects that are not on screen, and a pixel cannot refer to one. Use tier='plan' for the pixel channel.")
        self.refer = refer
        self.skills = tuple((PIXEL_SWAP.get(s, s) if refer == 'pixel' else s for s in self.tier.skills))
        self._allowed = frozenset(self.skills)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._allowed:
            raise TierError(f"{self.tier.key} tier has no skill {name!r}; it has {', '.join(sorted(self._allowed))}")
        if name in MANIPULATION:
            if self.refer == 'pixel':

                def pixel_verb(u, v, _v=name):
                    return self._sim.interact(_v, PixelRef(int(u), int(v)))
                return pixel_verb

            def verb(ref, into=None, _v=name):
                return self._sim.interact(_v, ref, into)
            return verb
        return getattr(self._sim, name)

    def __dir__(self):
        return sorted(self._allowed)

    def has(self, name):
        return name in self._allowed

    def prompt_lines(self):
        if self.refer == 'pixel':
            shape = {'go_to_object_at_pixel': 'go_to_object_at_pixel(u, v)', 'go_to_ground_at_pixel': 'go_to_ground_at_pixel(u, v)', 'move': 'move(dist_m)', 'strafe': 'strafe(dist_m)', 'turn': 'turn(deg)', 'look': 'look(deg)'}
            return tuple((shape.get(s, f'{s}(u, v)') for s in self.skills))
        shape = {'navigate_to_object': 'navigate_to_object(name)', 'go_to_visible_object': 'go_to_visible_object(name)', 'navigate_to_relative': 'navigate_to_relative(deg, dist_m)', 'move': 'move(dist_m)', 'strafe': 'strafe(dist_m)', 'turn': 'turn(deg)', 'look': 'look(deg)', 'place': 'place(name, into=name)'}
        return tuple((shape.get(s, f'{s}(name)') for s in self.skills))
