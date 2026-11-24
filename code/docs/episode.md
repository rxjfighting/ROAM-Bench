# Episode outputs

Each completed task writes a result row with its scene and task identifier, answer or goal correctness, N/S/T checks, charged-action count, and verification receipts. The aggregate result file also stores the model and effective evaluation configuration.

The task output directory retains the model conversation, executed actions and their feedback, exploration trace, visual-memory archive, and resumable JSON checkpoint. Raw model replies are preserved alongside the parsed action: the harness executes the first ACTION block rather than treating all generated text as an action.

The exploration trace records observations, object visibility, room coverage, and interaction events. Relocation events identify the source and destination, success, event order, held-object state, and destination-support readback. These evaluator records remain separate from the information sent to the model.

Use `python scripts/summarize.py ../results/model.json` to report A, N, S, and T rates, the correctness–task-completion gap, answer error among S-passing episodes, and average charged actions. Use `--resume` on the original evaluation command to continue matching results and interrupted episodes.

The result format contains local output paths because these are generated runtime artifacts. Credentials are read from the environment and are not saved in the result configuration.
