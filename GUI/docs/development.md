# Development

## Requirements

- Python 3.10 or newer;
- no third-party dependency for core and subprocess layers;
- PySide6 Essentials 6.8 or newer for the desktop interface.

## Principles

- Keep imports in `core` limited to the Python standard library.
- Keep validation separate from command construction.
- Do not invoke commands through a shell.
- Preserve `None` when it means "not specified by the user."
- Add a unit test whenever a CLI mapping or validation rule changes.
- Keep comments short; explain rationale in these Markdown files.

## Testing

Run from `GUI`:

```bash
python -m unittest discover -s tests -v
```

Core tests must not require Qt, CUDA, FFmpeg, models, or network access. Future
subprocess integration tests use short Python child processes rather than the
real pipeline.

PySide6 smoke tests run with Qt's offscreen platform and do not display a
window. They verify construction and mapping only; platform dialogs require
manual Windows and Linux checks.

The interface uses only QtCore, QtGui, and QtWidgets. Depend on
`PySide6-Essentials`, not the full PySide6 package, unless an interface feature
is added that requires an Addons module.

## Process events

The subprocess runner can be called from any interface library. Interface code
must drain its event queue on the GUI thread and should limit the number handled
per timer tick so rendering remains responsive. Events carry job IDs because a
runner may be reused while older events are still waiting to be consumed.

## Adding a CLI option

1. Update the neutral configuration model.
2. Add its deterministic mapping in the command builder.
3. Add relevant validation rules.
4. Update `cli-contract.md`.
5. Add or update tests.
6. Expose it through the interface adapter only after the core behavior is set.

## Important decisions requiring confirmation

Ask before changing the execution boundary, duplicating preset values, adding a
second GUI framework, persisting sensitive paths, supporting concurrent GPU
jobs, or changing the existing CLI.
