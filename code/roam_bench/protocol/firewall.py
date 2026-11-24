"""Firewall utilities for ROAM-Bench."""
import re
from typing import Optional
from .types import HELD_UNNAMED, ProposeResult, WorldView
_ARTICLES = re.compile('^(?:an?|the|some)\\s+', re.I)
_NONWORD = re.compile('[^a-z0-9]+')
VERDICT, NONE = ('verdict', 'none')

def normalise(text: str) -> str:
    t = _ARTICLES.sub('', str(text or '').strip().lower())
    return _NONWORD.sub('_', t).strip('_')

def kind_of(alias: str) -> str:
    return re.sub('_\\d+$', '', str(alias or ''))

class WorldModel:

    def __init__(self, *, partial_obs=True, furniture_prior=True, room_prior=True, strict_object_discovery=True, propose_feedback=VERDICT, include_relations=True, furniture_prior_kinds_only=False, world_text=True, hand_only=False):
        self.partial_obs = bool(partial_obs)
        self.furniture_prior = bool(furniture_prior)
        self.room_prior = bool(room_prior)
        self.strict = bool(strict_object_discovery)
        self.feedback = propose_feedback
        self.include_relations = bool(include_relations)
        self.furniture_prior_kinds_only = bool(furniture_prior_kinds_only)
        self.world_text = bool(world_text)
        self.hand_only = bool(hand_only)
        if self.hand_only and (not self.world_text):
            raise ValueError('hand_only=True needs world_text=True: it keeps one line of the text observation, so it cannot also be the switch that removes the text observation. For no text at all, pass world_text=False alone.')
        self._all_rooms = ()
        self._all_furniture = ()
        self._all_objects = ()
        self.seen_rooms = set()
        self.seen_furniture = set()
        self.discovered = set()
        self.labels = {}
        self.seen_relations = set()
        self.holding = None
        self.n_proposals = 0
        self.n_phantoms = 0

    def prime(self, *, rooms=(), furniture=(), objects=()):
        self._all_rooms = tuple(rooms)
        self._all_furniture = tuple(furniture)
        self._all_objects = tuple(objects)

    def observe(self, *, visible=(), rooms=(), relations=(), holding=None):
        visible = set(visible)
        self.holding = holding
        self.seen_rooms |= set(rooms)
        self.seen_furniture |= visible & set(self._all_furniture)
        if not self.strict:
            self.discovered |= visible & set(self._all_objects)
        if self.include_relations:
            self.seen_relations |= {(a, r, b) for a, r, b in relations}

    def known_rooms(self):
        if not self.partial_obs or self.room_prior:
            return set(self._all_rooms)
        return set(self.seen_rooms)

    def known_furniture(self):
        if not self.partial_obs or self.furniture_prior:
            return set(self._all_furniture)
        return set(self.seen_furniture)

    def furniture_for_view(self):
        known = self.known_furniture()
        if not (self.furniture_prior_kinds_only and self.partial_obs and self.furniture_prior):
            return known
        seen = set(self.seen_furniture) & known
        return seen | {kind_of(a) for a in known - seen}

    def known_objects(self):
        if not self.partial_obs:
            return set(self._all_objects)
        return set(self.discovered)

    def _display(self):
        if self.strict and self.feedback == NONE:
            return {alias: label for label, alias in self.labels.items() if alias}
        return {}

    def relations_for_view(self):
        show = self._display()
        known = self.known_objects() | self.furniture_for_view()
        out = set()
        for a, rel, b in self.seen_relations:
            if a not in known or b not in known:
                continue
            out.add((show.get(a, a), rel, show.get(b, b)))
        return out

    def view(self, here=None, visible=()) -> WorldView:
        if not self.world_text:
            return WorldView(blind=True)
        if self.hand_only:
            return WorldView(hand_only=True, holding=HELD_UNNAMED if self.holding else None)
        show = self._display()
        if self.strict and self.feedback == NONE:
            objects = tuple(sorted(self.labels))
        else:
            objects = tuple(sorted(self.known_objects()))
        furniture = tuple(sorted(self.furniture_for_view()))
        nameable = set(objects) | set(furniture)
        seen_now = tuple(sorted({show.get(a, a) for a in visible or ()} & nameable))
        return WorldView(rooms=tuple(sorted(self.known_rooms())), furniture=furniture, objects=objects, relations=tuple(sorted(self.relations_for_view())), holding=show.get(self.holding, self.holding), here=here or None, visible=seen_now)

    def propose(self, text, candidates=()) -> ProposeResult:
        self.n_proposals += 1
        label = normalise(text)
        if not label:
            return ProposeResult(False, None, 'Say what you think you see.')
        if self.feedback == NONE and self.labels.get(label) is not None:
            return ProposeResult(True, label, 'Noted.')
        same_kind = [c for c in candidates if kind_of(c) == label]
        fresh = next((c for c in same_kind if c not in self.discovered), None)
        if fresh is not None:
            self.discovered.add(fresh)
            granted = self._label_for(fresh, text)
            if self.feedback == NONE:
                return ProposeResult(True, granted, 'Noted.')
            return ProposeResult(True, granted, f'Yes -- you can refer to it as {granted}.')
        if same_kind:
            return ProposeResult(True, self._label_for(same_kind[0], text), 'Noted.')
        if self.feedback == NONE:
            self.labels.setdefault(label, None)
            self.n_phantoms += 1
            return ProposeResult(True, label, 'Noted.', phantom=True)
        return ProposeResult(False, None, "You are not close enough to anything like that, or it is not there. Get within arm's reach of what you want to name.")

    def _label_for(self, alias, text):
        if self.feedback == NONE:
            label = normalise(text)
            self.labels[label] = alias
            return label
        self.labels[alias] = alias
        return alias

    def resolve(self, label) -> Optional[str]:
        label = str(label or '')
        if not self.strict:
            if label in self.known_objects() or label in self.known_furniture():
                return label
            return None
        if label in self.labels:
            return self.labels[label]
        if label in self.known_furniture():
            return label
        return None

    def is_phantom(self, label) -> bool:
        return label in self.labels and self.labels[label] is None

    def is_shown_kind(self, label) -> bool:
        return label in self.furniture_for_view() and self.resolve(label) is None
