"""Discretize camera poses for task-required viewpoint coverage."""
from __future__ import annotations
import math
from typing import Mapping

POSITION_BIN_M = 0.5
YAW_BIN_DEG = 30.0
HORIZON_BIN_DEG = 15.0

def viewpoint_key(observation: Mapping) -> tuple | None:
    pose = observation.get('pose')
    room = observation.get('room_id')
    if not isinstance(pose, Mapping) or room is None:
        return None
    try:
        return (str(room), math.floor(float(pose['x']) / POSITION_BIN_M), math.floor(float(pose['z']) / POSITION_BIN_M), math.floor(float(pose['yaw_deg']) % 360.0 / YAW_BIN_DEG), math.floor(float(pose.get('horizon_deg', 0.0)) / HORIZON_BIN_DEG))
    except (KeyError, TypeError, ValueError):
        return None
