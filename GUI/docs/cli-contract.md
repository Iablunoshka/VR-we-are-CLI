# CLI Contract

This document describes the contract consumed by the GUI core. The authoritative
implementation remains `../main.py`.

## Modes

- `video`: input is a video file and output is a video file.
- `folder`: input is an image directory and output is a directory.
- `i2i`: input is an image or image directory; output shape follows the input.

## Option groups

- Required: `--input`, `--output`, `--input-type`.
- Processing: `--preset`, `--model`, `--batch-size`, `--autocast`,
  `--infer-accum-batches`.
- Video: `--codec`, `--quality`, and HDR options.
- Queues: `--r-queue`, `--in-queue`, `--p-queue`, `--s-queue`.
- Workers: `--feeders`, `--preprocess`, `--processors`, `--savers`.
- Conversion: `--depth-scale`, `--depth-offset`, `--switch-sides`,
  `--symmetric`, `--blur-radius`.
- Runtime: `--debug`, `--clean-output-pngs`.

## Rules represented in the core

- HDR is available only in video mode.
- HDR cannot be paired with an H.264 codec.
- Video mode requires one feeder and one saver when explicitly configured.
- Cleanup is available only in folder mode and cannot target the input folder.
- Presets are unavailable in i2i mode.
- Codec and quality are unavailable in i2i mode.
- Quality is unavailable in folder mode; codec is accepted but ignored with a
  warning to match current CLI behavior.
- Inference accumulation is available only in video and folder modes and must
  be at least one.
- A single i2i input requires a file output; directory i2i input requires a
  directory output when the output already exists.

## Compatibility policy

The command builder emits only public CLI arguments. Explicit settings override
preset values because that is the current CLI merge policy. Changes to argument
names, choices, or mode rules require synchronized tests and documentation.

