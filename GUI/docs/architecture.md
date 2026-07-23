# Architecture

## Goal

The GUI is an additional module that controls the existing CLI without importing
pipeline implementation details. Windows and Linux use the same core and PySide6
interface code.

## Layers

1. `core` contains configuration, validation, command construction, and job
   state. It has no GUI framework dependency.
2. The application layer coordinates validation, command preparation, starting,
   and cancelling a conversion.
3. The CLI adapter owns subprocess execution and translates process output into
   core job events.
4. The PySide6 layer renders configuration and CLI output and forwards user
   actions to the application layer.

Dependencies point inward: PySide6 and subprocess adapters may depend on the
core, while the core never imports either adapter.

## Initial decisions

### The CLI remains the execution boundary

The GUI will launch `main.py` as a child process. This keeps model, CUDA, FFmpeg,
and fatal pipeline errors isolated from the interface process.

### Commands are argument lists

`build_cli_command` returns an immutable sequence suitable for process APIs.
It does not create a shell command or perform quoting. This is safer and behaves
consistently on Windows and Linux.

### Validation has no side effects

Core validation reports structured errors and warnings. It does not create
output directories or modify files. The application layer may ask for user
confirmation before performing an action.

### Presets belong to the CLI

The core knows valid preset names but does not copy values from `presets.json`.
It passes the preset and explicit overrides to the CLI, which already defines
their merge behavior. This avoids two competing sources of truth.

### PySide6 is an adapter

No core type exposes Qt signals, widgets, models, or event-loop objects. Another
desktop interface library can be introduced by replacing the outer layer.
The adapter depends on PySide6 Essentials because it uses only QtCore, QtGui,
and QtWidgets. The full Addons wheel is intentionally excluded.

### Process execution uses the standard library

The CLI adapter uses `subprocess.Popen`, not `QProcess`. This keeps execution,
cancellation, and tests independent of Qt. The process is started without a
shell and stdout and stderr are consumed concurrently. The application starts
the Python CLI in UTF-8 mode so special characters have one explicit encoding
on Windows and Linux.

### Process events use a thread-safe queue

Reader threads publish typed events tagged with a job ID. An interface drains
the queue on its own main-thread timer. This prevents worker threads from
calling widgets and supports Qt, Tk, web, or headless consumers. The queue is
currently unbounded so a slow interface cannot block the CLI output pipes;
output coalescing may be introduced if real workloads produce excessive logs.

### Cancellation is staged

The runner first requests a graceful stop for the process group, waits for a
short grace period, and then forces termination. The adapter reports the job as
cancelled once the child exits. Deleting or retaining partial output remains an
application-policy decision.

### CLI output is the progress display

The first interface does not infer numerical progress from unstable text. It
shows stdout and stderr as produced by the CLI and uses process events only for
the high-level status and button state. Carriage-return records are preserved as
transient events, allowing FFmpeg statistics to redraw one line as they do in a
terminal.

### Successful completion has a temporary visual signal

The status label flashes green for seven seconds after a successful conversion,
then returns to the normal `Ready` state. The effect is presentation-only and
does not change job state or rely on an application-wide stylesheet.

### CLI location is configurable

Source runs locate `main.py` in the parent project and use the active Python
executable. Environment overrides provide an explicit path boundary for other
development layouts and future packaging.

### The desktop interface follows the system theme

Qt follows the Windows and Linux color-scheme hint. The interface does not set
an application-wide stylesheet or explicit color palette, allowing palette
changes to propagate to every widget. Typography uses widget font properties,
which do not interfere with system theme updates.

### File pickers keep separate persistent state

Input and output dialogs remember independent directories across application
restarts. This UI-only preference uses Qt's platform settings store and does
not enter the conversion core. Missing saved directories fall back to the
user's home directory.

### Optional settings are progressively disclosed

Stereo conversion, video, HDR, and advanced pipeline controls are separate
collapsible sections that start closed. Collapsing a section affects
presentation only: configured values remain active and are not reset. Advanced
overrides retain a separate enable checkbox, so opening the section does not
apply them. This behavior belongs entirely to the interface adapter.
