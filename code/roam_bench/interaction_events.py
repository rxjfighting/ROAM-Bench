"""Link interaction receipts to decision rounds and charged action numbers."""
from copy import deepcopy


INTERACTION_VERBS = frozenset({"pick", "place", "open", "close", "toggle"})


def stamp_interaction_events(events, start, observation_round, action_step):
    """Add harness coordinates without changing the backend's internal counter."""
    for index in range(start, len(events)):
        events[index].update(event_index=index + 1,
                             observation_round=observation_round,
                             action_step=action_step)


def aligned_interaction_events(record, verification):
    """Validate the ordered action/event correspondence and add display indices.

    Backend step_idx is not a charged-action count. In particular, interactions
    and some rejected navigation attempts do not advance it. All charged
    actions, including unsuccessful attempts, participate in action_step.
    """
    actions = []
    charged = 0
    for row in record["rounds"]:
        before = row.get("current_pose", {}).get("action_steps")
        if before is not None and before != charged:
            raise ValueError("Round pose disagrees with the charged-action count")
        execution = row.get("execution")
        if bool(execution) != bool(row["charged_action"]):
            raise ValueError("Charged action lacks a corresponding execution")
        if not row["charged_action"]:
            continue
        charged += 1
        if row.get("action_step", charged) != charged:
            raise ValueError("Round action step disagrees with the execution order")
        if execution["skill"] != row["model_output"]["parsed_action"]["skill"]:
            raise ValueError("Model action and execution skill disagree")
        if execution["skill"] in INTERACTION_VERBS:
            actions.append((row, charged))
    if charged != record["charged_actions"]:
        raise ValueError("Episode charged-action total disagrees with its rounds")
    events = deepcopy(verification.get("interaction_events", []))
    if len(events) != len(actions):
        raise ValueError("Interaction event count disagrees with executed actions")
    for index, (event, (row, action_step)) in enumerate(zip(events, actions), 1):
        execution = row["execution"]
        if (event["verb"] != execution["skill"] or
                bool(event["success"]) != bool(execution["success"])):
            raise ValueError("Interaction event disagrees with its executed action")
        coordinates = {"event_index": index, "observation_round": row["round"],
                       "action_step": action_step}
        for key, value in coordinates.items():
            if key in event and event[key] != value:
                raise ValueError(f"Interaction {key} disagrees with its executed action")
        event.update(coordinates)
    return events
