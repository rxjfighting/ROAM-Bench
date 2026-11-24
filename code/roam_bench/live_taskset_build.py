"""Live taskset build utilities for ROAM-Bench."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
from roam_bench.protocol.types import NameRef

@dataclass(frozen=True)
class NavigationMeasurements:
    room_graph: Mapping[str, tuple[str, ...]]
    standing_room: Mapping[str, int]
    reachable_by_legal_navigation: Mapping[str, bool] | None = None

    def generator_options(self) -> dict:
        graph = {room: tuple(neighbours) for room, neighbours in self.room_graph.items()}
        standing = {alias: int(count) for alias, count in self.standing_room.items()}
        reach_options = {'standing': standing}
        if self.reachable_by_legal_navigation is not None:
            reach_options['reachable'] = {alias: bool(value) for alias, value in self.reachable_by_legal_navigation.items()}
        return {'topo_route': {'graph': graph, 'standing': standing}, 'reach_blocked': reach_options}

    def to_dict(self) -> dict:
        out = {'room_graph': {room: list(neighbours) for room, neighbours in sorted(self.room_graph.items())}, 'standing_room': {alias: int(count) for alias, count in sorted(self.standing_room.items())}, 'room_count': len(self.room_graph), 'edge_count': sum((len(neighbours) for neighbours in self.room_graph.values())) // 2, 'standing_alias_count': len(self.standing_room), 'blocked_alias_count': sum((int(count) == 0 for count in self.standing_room.values()))}
        if self.reachable_by_legal_navigation is not None:
            out['reachable_by_legal_navigation'] = {alias: bool(value) for alias, value in sorted(self.reachable_by_legal_navigation.items())}
            out['legal_navigation_probe_count'] = len(self.reachable_by_legal_navigation)
        return out

class _ProbeRecorder:

    def __init__(self):
        self.steps = []

    def step(self, kind, arg, result):
        self.steps.append((kind, arg, bool(result.success), result.reason))
        return result

def measure_legal_reachability(sim, measured: NavigationMeasurements, aliases, *, reset: Callable[[], None] | None=None) -> tuple[NavigationMeasurements, dict[str, dict]]:
    from .legal_oracle import LegalOracleSim
    outcomes, receipts = ({}, {})
    for alias in sorted(set(map(str, aliases))):
        if reset is not None:
            reset()
        recorder = _ProbeRecorder()
        proxy = LegalOracleSim(sim, recorder, tier='spatial', refer='pixel')
        result = proxy.navigate_to_object(NameRef(alias))
        outcomes[alias] = bool(result.success)
        receipts[alias] = {'success': bool(result.success), 'reason': str(result.reason), 'public_steps': len(recorder.steps), 'moved_m': float(getattr(result, 'moved_m', 0.0)), 'collided': bool(getattr(result, 'collided', False))}
    return (NavigationMeasurements(room_graph=measured.room_graph, standing_room=measured.standing_room, reachable_by_legal_navigation=outcomes), receipts)

def measure_navigation(sim) -> NavigationMeasurements:
    room_graph = getattr(sim, 'room_graph', None)
    standing_room = getattr(sim, 'standing_room', None)
    if not callable(room_graph) or not callable(standing_room):
        raise TypeError('live navigation generation requires room_graph() and standing_room()')
    graph = dict(room_graph())
    standing = dict(standing_room())
    if not graph:
        raise RuntimeError('production room_graph measurement returned no rooms')
    if not standing:
        raise RuntimeError('production standing_room measurement returned no aliases')
    unknown = {neighbour for neighbours in graph.values() for neighbour in neighbours} - set(graph)
    if unknown:
        raise RuntimeError(f'production room_graph contains unknown neighbours: {sorted(unknown)}')
    return NavigationMeasurements(room_graph={room: tuple(sorted(neighbours)) for room, neighbours in sorted(graph.items())}, standing_room={alias: int(count) for alias, count in sorted(standing.items())})
