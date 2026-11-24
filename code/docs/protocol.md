# Evaluation protocol

## Model inputs and budgets

The main protocol sends a 768 × 432 RGB observation, a cumulative self-odometry plot, and action/hand/contact feedback. RGB images use JPEG quality 90. Odometry contains the robot's own positions, headings and step order, with no room map or object locations. Task references follow the captions in the task set.

Recent visual history keeps 20 rounds; earlier text and selected images remain in the archive. A decision has at most 50 image parts, including references, recent history, intermediate candidates, retrieval, odometry and current RGB. An action can produce up to eight intermediate candidates at 1 m translation or 30° rotation spacing. The model selects retained candidates on its first subsequent decision; unselected candidates do not remain retrievable. Current final RGB supplies action pixel coordinates.

Up to two retrieval passes occur before an action or answer, with at most 12 retrieved images in total per decision. Retrieval itself is not charged as a robot action. The task set records each episode's action budget, normally 25 times its room count. Navigation, viewpoint changes and interaction skills consume charged actions; the final answer does not. The harness permits eight extra decision rounds and terminates after eight accumulated malformed replies or when its episode limits are reached.

## Model requests

Evaluation uses temperature 0, top-p 1, and the highest reasoning effort supported by each model. The client defaults to `reasoning_effort=max`; for an endpoint with a different name for its highest level, pass that value with `--reasoning-effort`. A rejected reasoning setting terminates the call rather than silently falling back to a lower effort. Transport permits ten transient attempts per call with a 120 s timeout per attempt. Completed responses are accepted without resampling for answer quality.

The interface first sends native message roles. If an endpoint explicitly refuses an image's role, an adapter moves only those image parts to user-role messages while preserving all text roles and the full content order. Explicitly refused optional request fields can also be adapted. Unsupported vision, authentication and permanent format errors terminate the call. The adapter does not substitute models, endpoints or credentials. Its statistics record applied adaptations.

The harness parses the first ACTION block from a reply and records the entire raw response, including text after that block. Malformed or unavailable actions receive an explicit error; a denied or failed skill cannot count as a successful state change. See [episode outputs](episode.md) for the saved record structure.

## Independent metrics

- **A**: answer or relocation outcome correctness under the task's rule.
- **N**: coverage of task-required observations and interactions, including ordered events where specified.
- **S**: N together with stronger task-specific corroboration, such as geometric readbacks or verified destination support after placement.
- **T (strict task success)**: A and N both pass, measuring instruction-conditioned embodied task completion.

The correctness–task-completion gap is the fraction of all episodes with A=1 and N=0, equal to the difference between A and T rates. It covers incomplete object observations, search coverage, or prescribed procedures. Path-integration landmark shortfalls are reported separately from target-observation shortfalls because self-odometry provides direction information.

N conditions operationalize observations and interactions specified by each task instruction and are fixed before model evaluation. N separately checks relevant object observations, search completeness, and prescribed exploration or interaction. For path integration, the frozen landmark check requires observation of both instructed landmarks, regardless of their requested sequence. Self-odometry also provides direction information. Relocation checks pick-before-place order for the same accepted object. Whole-house counting checks the declared search scope alongside observed target instances. Missing telemetry fails verification. N/S are explicit benchmark conditions; they do not establish that a model's internal reasoning depends on the observations, nor that every covered observation is perceptually sufficient. S means stronger corroboration under the implemented task rule.

For relocation, N requires a successful pick of the accepted source followed by successful placement of that same source onto the destination. S additionally requires a post-placement support readback matching the designated destination. When a task accepts multiple sources, events from different sources cannot be pooled.

Each task saves correctness, N/S/T status, detailed verification receipts, the full model rounds, exploration trace, and a resumable state checkpoint. `summarize.py` reports metric counts/rates and answer error among S-passing episodes.

## Motion video rendering

Video frames use at least 16 rendering subframes at each camera pose to let real-time lighting and denoising settle after motion. `backend.set_film_quality(rt_subframes=16)` controls this minimum independently of observation rendering. It preserves the action path and output frame rate; no color correction is applied to recorded observations.

`backend.set_film_rate(fps=24)` doubles the number of rendered motion poses relative to 12 FPS and scales frame holds to preserve playback duration. The video writer must use the same frame rate. This changes video sampling only; the observations delivered to the model and the task checks are unaffected.
