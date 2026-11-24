"""Feedback utilities for ROAM-Bench."""
import re
VERDICT, NONE, SENSOR = ('verdict', 'none', 'sensor')
MODES = (VERDICT, NONE, SENSOR)
WITHHELD = 'Your last action was submitted to the robot. What it did is not reported to you -- read the new photograph.'
UNMEASURED = 'Your last action was submitted to the robot. Its own sensors report nothing conclusive -- read the new photograph.'
CMD = 'cmd'
LIMIT = 'limit'
REACH = 'reach'
NO_EFFECT = 'no_effect'
HAND_EMPTY = 'hand_empty'
HAND_FULL = 'hand_full'
SURFACE = 'surface'
NO_RANGE = 'no_range'
NO_ROUTE = 'no_route'
MOTION = 'motion'
DID = 'did'
UNKNOWN = 'unknown'
_VERB_REFUSAL = re.compile('^you cannot \\S+ things\\b')
_SENSE_RULES = (('not a point in your view', CMD), ('outside your view', CMD), ('cannot refer to a thing that way', CMD), ('is a direction, not an object', CMD), (_VERB_REFUSAL, CMD), ('change your view by 0', CMD), ('do not need to name things in this task', CMD), ('that point is empty space', NO_RANGE), ('cannot work out where that point is', NO_RANGE), ('nothing you can interact with at that point', NO_RANGE), ('floor, wall or ceiling', SURFACE), ('part of the room', SURFACE), ('an object, not ground', SURFACE), ('above you, not ground', SURFACE), ('not something you can handle', SURFACE), ('nowhere to stand', NO_ROUTE), ('no walkable ground', NO_ROUTE), ('cannot find a way to stand next to it', NO_ROUTE), ('cannot find a way to get there', NO_ROUTE), ('hand is already full', HAND_FULL), ('not holding anything', HAND_EMPTY), ('hand was already empty', HAND_EMPTY), ('not enough room to lift it', REACH), ('cannot fit it there from this angle', REACH), ('too far away to reach', REACH), ('looking as far', LIMIT), ('as far as you can go', LIMIT), ('could not change where you are looking', LIMIT), ('you walked', MOTION), ('you turned', MOTION), ('you looked', MOTION), ('could not turn', MOTION), ('could not move', MOTION), ('still facing the same way', MOTION), ('you picked it up', DID), ('you put it down there', DID), ('you opened it', DID), ('you closed it', DID), ('you switched it', DID), ('you sliced it', DID), ('you broke it', DID), ('need to open it first', NO_EFFECT), ('not looking at it', NO_EFFECT), ('too far away, or something is between', NO_EFFECT), ('not something you can pick up', NO_EFFECT), ('does not open', NO_EFFECT), ('does not switch on or off', NO_EFFECT), ('does not go there', NO_EFFECT), ('not somewhere you can put things', NO_EFFECT), ('cannot see that from where you are standing', NO_EFFECT), ('too far away from here', NO_EFFECT), ('too little of it is in view', NO_EFFECT), ('nothing there', NO_EFFECT), ('you know this room has one', NO_EFFECT), ('have not found anything by that name', NO_EFFECT), ('do not know of anything by that name', NO_EFFECT), ('already ', NO_EFFECT), ('did not work', UNKNOWN))

def sense_of(reason):
    low = str(reason or '').lower()
    for needle, code in _SENSE_RULES:
        if hasattr(needle, 'search'):
            if needle.search(low):
                return code
        elif needle in low:
            return code
    return UNKNOWN
_NOT_VISIBLE_MARKS = ('cannot see', 'too far away from here', 'too little of it is in view')

def not_visible_here(reason):
    low = str(reason or '').lower()
    return any((m in low for m in _NOT_VISIBLE_MARKS))

def _moved_of(result):
    return abs(float(getattr(result, 'moved_m', 0.0) or 0.0))

def _odometry(result):
    turned = float(getattr(result, 'turned_deg', 0.0) or 0.0)
    moved = float(getattr(result, 'moved_m', 0.0) or 0.0)
    parts = []
    if abs(turned) >= 0.5:
        parts.append(f"turned {abs(turned):.0f}° to the {('left' if turned > 0 else 'right')}")
    if abs(moved) >= 0.005:
        parts.append(f'travelled {moved:.2f} m')
    if not parts:
        parts.append('did not move')
    return ', then '.join(parts)
_LOOK = ' Whether that achieved what you wanted is not something the robot can feel -- read the new photograph.'

def _sense_base(result, ok, sense, verb):
    line = f'Odometry: the base {_odometry(result)}'
    asked = getattr(result, 'requested_m', None)
    moved = _moved_of(result)
    if asked is not None and moved >= 0.005 and (abs(float(asked)) - moved > 0.02):
        line += f' of the {abs(float(asked)):.2f} m commanded'
    line += '.'
    if getattr(result, 'collided', False):
        line += ' The drive current spiked against an obstacle.'
    return line

def _sense_look(result, ok, sense, verb):
    if ok:
        if sense == LIMIT:
            return 'Head encoder: the camera pitched as commanded and the joint is now at the end of its travel in that direction.'
        return 'Head encoder: the camera pitch changed by the angle you commanded.'
    if sense == LIMIT:
        return 'Head encoder: the camera did not move -- the pitch joint is already at the end of its travel in that direction.'
    return UNMEASURED

def _sense_goto(result, ok, sense, verb):
    moved = float(getattr(result, 'moved_m', 0.0) or 0.0)
    turned = float(getattr(result, 'turned_deg', 0.0) or 0.0)
    if abs(moved) >= 0.005 or abs(turned) >= 0.5:
        line = f'Odometry: the base {_odometry(result)}, and '
        if ok:
            line += 'came to rest at the pose the planner was driving to.'
        elif getattr(result, 'collided', False):
            line += 'then stalled against an obstacle before reaching that pose -- the drive current spiked.'
        else:
            line += 'stopped short of that pose.'
        return line
    pixel = verb in ('go_to_object_at_pixel', 'go_to_ground_at_pixel')
    if sense == NO_RANGE:
        return 'Depth camera: no usable range return at that pixel, so the planner had no 3-D goal to drive to. The base did not move.'
    if sense == SURFACE:
        which = 'stands above the ground plane rather than on it, so it is not a place the base could stand' if verb == 'go_to_ground_at_pixel' else 'lies on the ground plane itself, so there is no raised body there to drive up to'
        line = f'Depth camera: the point behind that pixel {which}. The base did not move.'
        if pixel:
            line += ' The other navigation skill is the one that takes points of that kind.'
        return line
    if sense == NO_ROUTE:
        return 'Local map: no free standing pose near that goal, so the planner produced no path. The base did not move.'
    return 'Odometry: the base did not move. The planner rejected that goal before commanding the wheels, and its own sensors do not say what was wrong with it.'

def _hand_fail(sense, verb):
    if sense == REACH:
        return 'Arm: the inverse-kinematics solver found no pose that reaches that point from where the base is standing. The arm did not move.'
    if sense == NO_RANGE:
        return 'Depth camera: no usable range return at that pixel, so there was no 3-D target for the hand. The arm did not move.'
    if sense == SURFACE:
        return 'Depth camera: the point behind that pixel lies on the room shell -- a floor, wall or ceiling plane -- not on a body standing free of it. The arm did not move.'
    if sense == HAND_FULL:
        return 'Gripper: it is already closed on something.'
    if sense == HAND_EMPTY:
        return 'Gripper: it reports nothing between the fingers.'
    return None

def _sense_pick(result, ok, sense, verb):
    if ok:
        return 'Gripper: the fingers closed on an object and are now holding it.'
    if sense == HAND_FULL:
        return 'Gripper: it is already closed on something, and it has only one set of fingers. Put that down first.'
    if sense == NO_EFFECT:
        return 'Gripper: the fingers closed and came back empty.'
    return _hand_fail(sense, verb) or UNMEASURED

def _sense_place(result, ok, sense, verb):
    if ok:
        return 'Gripper: the fingers opened and the wrist no longer carries a load, so the object has left the hand.' + _LOOK
    if sense == HAND_EMPTY:
        return 'Gripper: there was nothing between the fingers to release.'
    if sense == NO_EFFECT:
        return 'Gripper: the fingers did not open -- the wrist still carries the load, so the object is still in the hand.'
    return _hand_fail(sense, verb) or UNMEASURED

def _sense_hinge(result, ok, sense, verb):
    if ok:
        return 'Wrist force sensor: the hand met resistance and the target displaced under it until it came to a stop.'
    if sense == NO_EFFECT:
        return 'Wrist force sensor: the hand arrived and loaded up against the target, and nothing displaced. It did not move at all.'
    return _hand_fail(sense, verb) or UNMEASURED

def _sense_toggle(result, ok, sense, verb):
    if ok:
        return 'Wrist force sensor: the hand met the target and something clicked over under it.' + _LOOK
    if sense == NO_EFFECT:
        return 'Wrist force sensor: the hand arrived and loaded up against the target, and nothing clicked or displaced.'
    return _hand_fail(sense, verb) or UNMEASURED
_SENSOR = {'turn': _sense_base, 'move': _sense_base, 'strafe': _sense_base, 'look': _sense_look, 'navigate_to_object': _sense_goto, 'go_to_visible_object': _sense_goto, 'navigate_to_relative': _sense_goto, 'go_to_object_at_pixel': _sense_goto, 'go_to_ground_at_pixel': _sense_goto, 'pick': _sense_pick, 'place': _sense_place, 'open': _sense_hinge, 'close': _sense_hinge, 'toggle': _sense_toggle}

class Feedback:

    def __init__(self, action_feedback=VERDICT):
        if action_feedback not in MODES:
            raise ValueError('action_feedback must be one of ' + ', '.join((repr(m) for m in MODES)) + f', got {action_feedback!r}')
        self.mode = action_feedback

    @classmethod
    def from_episode(cls, episode):
        return cls(getattr(episode, 'action_feedback', VERDICT))

    @property
    def withholds(self):
        return self.mode == NONE

    @property
    def senses(self):
        return self.mode == SENSOR

    def for_action(self, result, verb=None):
        if self.withholds:
            return WITHHELD
        if self.senses:
            return self._sensor(result, verb)
        return ('OK. ' if result.success else 'REFUSED. ') + (getattr(result, 'reason', '') or '')

    def _sensor(self, result, verb):
        ok = bool(getattr(result, 'success', False))
        sense = sense_of(getattr(result, 'reason', ''))
        if sense == CMD and (not ok):
            return 'REJECTED before anything moved: ' + (getattr(result, 'reason', '') or '')
        handler = _SENSOR.get(str(verb or '').strip().lower())
        if handler is None:
            return UNMEASURED
        return handler(result, ok, sense, str(verb or '').strip().lower())

    def for_refusal(self, why):
        return 'REFUSED before reaching the simulator: ' + (why or '')

    def for_parse_error(self, why):
        return 'Your reply could not be used: ' + (why or '') + '\nReply with exactly two lines, THOUGHT: and ACTION:, and put one call from the skill list on the ACTION line.'
