# VR We Are GUI

This directory contains the independent GUI module for VR We Are. Its PySide6
desktop interface is a thin layer over a framework-neutral core and subprocess
adapter.

The existing CLI remains the execution backend and source of truth for pipeline
defaults and presets.

## Current scope

- typed conversion configuration;
- side-effect-free validation;
- deterministic CLI argument construction;
- framework-neutral job states and results;
- asynchronous subprocess execution through a thread-safe event queue;
- a Windows/Linux PySide6 desktop interface;
- unit tests for the core behavior.

## Structure

```
pyproject.toml                 Package metadata, GUI extra, and console entry point
README.md                      Setup, structure, and usage overview

vr_we_are_gui/
  __main__.py                  Supports `python -m vr_we_are_gui`
  core/
    config.py                  Framework-neutral conversion configuration and enums
    validation.py              Side-effect-free configuration validation
    command.py                 Deterministic CLI argument construction
    jobs.py                    Job states, results, and process event types
  application/
    controller.py              Prepare, start, cancel, and drain conversion operations
  adapters/
    subprocess_runner.py       CLI process execution, cancellation, and output streaming
  interfaces/
    pyside6/
      app.py                   PySide6 startup and CLI target discovery
      main_window.py           PySide6 controls, dialogs, and event rendering
      theme.py                 System-theme-compatible appearance setup

tests/
  test_core.py                 Configuration, validation, command, and job-state tests
  test_application.py          Controller behavior and runner contract tests
  test_subprocess_runner.py    Process execution, cancellation, and output tests
  test_pyside.py               PySide6 interface smoke and behavior tests

docs/
  architecture.md              Layer boundaries and design decisions
  development.md               Development workflow and CLI-option checklist
  cli-contract.md              CLI arguments and GUI compatibility rules
```

Only `interfaces/pyside6` depends on PySide6. The core, application, and
subprocess adapter can be reused by a Tkinter, Gradio, or other interface.

## Run the interface

Install the interface dependency into the active environment:

```bash
python -m pip install -e ".[pyside6]"
```

Then run from this directory:

```bash
python -m vr_we_are_gui
```

By default, the interface uses the active Python executable and expects the CLI
in the parent directory. `VR_WE_ARE_PYTHON` and `VR_WE_ARE_CLI_ROOT` can override
those locations.

## Run the tests

From this directory, using the project environment:

```powershell
..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux, with an activated environment:

```bash
python -m unittest discover -s tests -v
```

## Documentation

- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [CLI contract](docs/cli-contract.md)

