"""Policies utilities for ROAM-Bench."""
import math
import random
from dataclasses import dataclass
from .feedback import not_visible_here
from .types import DirectionRef, NameRef, PixelRef
_SEARCH_TURN = 45.0
_SEARCH_MAX = 8
_APPROACH_ROUNDS = 1
_ADVANCE_M = 1.5
_STOP_AT_M = 0.9
_LEG_M = 1.0
_SPATIAL_MAX_LEGS = 10

@dataclass(frozen=True)
class Perception:
    bearing_sigma_deg: float = 0.0
    distance_sigma_m: float = 0.0
    click_sigma_px: float = 0.0
    bearing_bias_deg: float = 0.0
    distance_bias_m: float = 0.0
    seed: int = 0

    @property
    def label(self):
        bits = []
        if self.bearing_sigma_deg:
            bits.append(f'±{self.bearing_sigma_deg:g}°')
        if self.distance_sigma_m:
            bits.append(f'±{self.distance_sigma_m:g}m')
        if self.click_sigma_px:
            bits.append(f'±{self.click_sigma_px:g}px')
        if self.bearing_bias_deg:
            bits.append(f'{self.bearing_bias_deg:+g}°偏')
        if self.distance_bias_m:
            bits.append(f'{self.distance_bias_m:+g}m偏')
        return '/'.join(bits) if bits else 'oracle'

    def bearing(self, truth, rng):
        if truth is None:
            return None
        deg, dist = truth
        deg += self.bearing_bias_deg
        dist += self.distance_bias_m
        if self.bearing_sigma_deg:
            deg += rng.gauss(0.0, self.bearing_sigma_deg)
        if self.distance_sigma_m:
            dist += rng.gauss(0.0, self.distance_sigma_m)
        return (deg, max(0.0, dist))

    def click(self, px, rng):
        if px is None or not self.click_sigma_px:
            return px
        u, v = px
        return (int(round(u + rng.gauss(0.0, self.click_sigma_px))), int(round(v + rng.gauss(0.0, self.click_sigma_px))))

class TieredOracle:

    def __init__(self, sim, space, recorder, perception=Perception(), channel='name', on_step=None):
        self.sim = sim
        self.space = space
        self.rec = recorder
        self.per = perception
        self.channel = channel
        self.rng = random.Random(perception.seed)
        self.on_step = on_step
        self.budget = recorder.task.max_steps
        self.notes = []

    def _act(self, kind, arg, fn, note=''):
        if len(self.rec.steps) >= self.budget:
            return None
        res = fn()
        self.rec.step(kind, arg, res)
        self.notes.append(note or kind)
        if self.on_step:
            self.on_step(self, kind, arg, res, note)
        return res

    def _ref(self, alias):
        if self.channel == 'name':
            return (NameRef(alias), None)
        px = self.sim.instance_centroid(alias)
        if px is None and self.space.has('turn'):
            est = self.per.bearing(self.sim.bearing_to(alias), self.rng)
            if est and abs(est[0]) > 5:
                self._act('turn', est[0], lambda d=est[0]: self.space.turn(d), '转向目标才能点到它')
            px = self.sim.instance_centroid(alias)
        if px is None:
            return (None, '目标不在画面里，点不到')
        noisy = self.per.click(px, self.rng)
        return (PixelRef(*noisy), None)

    def go(self, alias):
        key = self.space.tier.key
        if key == 'high':
            return self._go_high(alias)
        if key == 'plan':
            return self._go_plan(alias)
        return self._go_spatial(alias)

    def _go_high(self, alias):
        return self._act('navigate_to_object', NameRef(alias), lambda: self.space.navigate_to_object(NameRef(alias)), '按名字直达（技能替大脑找到了它）')

    def _go_plan(self, alias):
        for round_i in range(_APPROACH_ROUNDS + 1):
            res = self._sweep(alias, round_i)
            if res is None or res.success:
                return res
            if not not_visible_here(res.reason):
                return res
            if round_i == _APPROACH_ROUNDS:
                return res
            if self._act('navigate_to_relative', DirectionRef(0.0, _ADVANCE_M), lambda: self.space.navigate_to_relative(0.0, _ADVANCE_M), '转了一圈都看不见，往前挪一点再找') is None:
                return None
        return res

    def _sweep(self, alias, round_i=0):
        res = None
        for i in range(_SEARCH_MAX):
            if i:
                if self._act('turn', _SEARCH_TURN, lambda: self.space.turn(_SEARCH_TURN), '看不见它，转身继续找') is None:
                    return None
            note = '尝试走向可见目标' if i == 0 and (not round_i) else f'第 {round_i + 1} 圈第 {i + 1} 次尝试'
            res = self._act('go_to_visible_object', NameRef(alias), lambda: self.space.go_to_visible_object(NameRef(alias)), note)
            if res is None or res.success or (not not_visible_here(res.reason)):
                return res
        return res

    def _go_spatial(self, alias):
        last = None
        for _ in range(_SPATIAL_MAX_LEGS):
            est = self.per.bearing(self.sim.bearing_to(alias), self.rng)
            if est is None:
                return None
            deg, dist = est
            if dist <= _STOP_AT_M:
                return last
            if abs(deg) > 5.0:
                last = self._act('turn', deg, lambda d=deg: self.space.turn(d), f'估计目标在左 {deg:+.0f}°')
                if last is None:
                    return None
            leg = max(0.25, min(_LEG_M, dist - _STOP_AT_M * 0.8))
            last = self._act('move', leg, lambda d=leg: self.space.move(d), f'估计还差 {dist:.2f} m，先走 {leg:.2f} m')
            if last is None:
                return None
            if last.collided:
                if self._act('turn', 45.0, lambda: self.space.turn(45.0), '撞停了，侧开一点再重新估计') is None:
                    return None
        return last

    def manipulate(self, verb, alias):
        ref, why = self._ref(alias)
        if ref is None:
            self.notes.append(why)
            return None
        fn = getattr(self.space, verb)
        return self._act(verb, ref, lambda: fn(ref), f'{verb} 目标')

    def run(self, plan):
        for verb, alias in plan:
            if verb == 'go':
                self.go(alias)
            else:
                self.manipulate(verb, alias)
            if len(self.rec.steps) >= self.budget:
                break
        return self.rec

def run_plan(sim, task, plan, tier, perception=Perception(), channel='name', on_step=None, dilation=2):
    from .score import Recorder
    from .tiers import ActionSpace
    space = ActionSpace(sim, tier)
    rec = Recorder(sim, task, dilation=dilation)
    pol = TieredOracle(sim, space, rec, perception, channel, on_step)
    pol.run(plan)
    return (rec, pol)

def tier_tax(rows, metric='success_rate'):
    out = {}
    keys = list(rows)
    for i in range(len(keys) - 1):
        a, b = (keys[i], keys[i + 1])
        x, y = (rows[a].get(metric), rows[b].get(metric))
        out[f'{a}-{b}'] = None if x is None or y is None else x - y
    return out

def bearing_error_after(deg_sigma, dist_m):
    return abs(math.tan(math.radians(deg_sigma))) * dist_m
