# Sub-task VLA dataset conversion

This directory converts labeled intervals from a Breakfast LeRobot v2.1
dataset into a sub-task VLA dataset.

Each labeled sub-task interval becomes an independent LeRobot episode. Frames
outside `split/*/sub_task` are omitted. Splitting at every sub-task boundary is
intentional: it prevents an action horizon conditioned on one instruction from
crossing into the next instruction.

The output keeps:

- all three camera views;
- the original 30 Hz sampling rate;
- all state and action columns;
- half-open annotation intervals `[start_frame, end_frame)`.

It rewrites `timestamp`, `frame_index`, `episode_index`, global `index`, and
`task_index`, and regenerates all LeRobot metadata. `source_mapping.jsonl`
provides complete provenance for every output episode.

## Commands

Use the Python environment bundled with OpenPI (it already contains PyArrow,
LeRobot, and the required video packages):

```bash
PYTHON=/home/geek/share3/vla-fusion/geomodel/openpi/.venv/bin/python

# Small end-to-end test
$PYTHON vla_data_process/convert_subtask_vla.py \
  --source /home/geek/share3/agilex_make_breakfast_380 \
  --split-dir /path/to/trajectory_splitter_output/split \
  --task-spec vla_data_process/agilex_make_breakfast.json \
  --output /home/geek/share3/agilex_make_breakfast_subtask_vla_test \
  --max-source-episodes 1

# Full conversion
$PYTHON vla_data_process/convert_subtask_vla.py \
  --source /home/geek/share3/agilex_make_breakfast_380 \
  --split-dir /path/to/trajectory_splitter_output/split \
  --task-spec vla_data_process/agilex_make_breakfast.json \
  --output /home/geek/share3/agilex_make_breakfast_subtask_vla \
  --workers 2

# Resume an interrupted conversion and increase parallelism
$PYTHON vla_data_process/convert_subtask_vla.py \
  --source /home/geek/share3/agilex_make_breakfast_380 \
  --split-dir /path/to/trajectory_splitter_output/split \
  --task-spec vla_data_process/agilex_make_breakfast.json \
  --output /home/geek/share3/agilex_make_breakfast_subtask_vla \
  --workers 8 --resume

# Independent structural and decode validation
$PYTHON vla_data_process/validate_subtask_vla.py \
  /home/geek/share3/agilex_make_breakfast_subtask_vla \
  --decode-samples 16
```

The converter refuses to overwrite an existing output directory. Remove or
rename an unwanted test output explicitly before rerunning it.

## Statistics note

Vector statistics are recomputed exactly for every cropped interval. Image
statistics are copied from the corresponding source episode with their sample
counts adjusted to the cropped length. This is sufficient for LeRobot v2.1
metadata loading; OpenPI computes and uses its own normalization statistics for
state and actions, and does not normalize camera pixels from these metadata
statistics.
