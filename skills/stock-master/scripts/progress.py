"""
Progress indicator utilities for long-running operations.

Provides:
- TTY-aware progress indicators (spinners for interactive, dots for non-interactive)
- Step-based progress tracking
- Context manager for automatic cleanup
"""

import sys
import threading
import time
from contextlib import contextmanager


class ProgressIndicator:
    """Progress indicator with TTY detection."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str, use_spinner: bool = None):
        """
        Initialize progress indicator.

        Args:
            message: Message to display
            use_spinner: Force spinner on/off (auto-detected if None)
        """
        self.message = message
        self.is_tty = sys.stdout.isatty() if use_spinner is None else use_spinner
        self.running = False
        self.thread: threading.Thread | None = None
        self._frame_index = 0

    def start(self):
        """Start the progress indicator."""
        if self.running:
            return

        self.running = True

        if self.is_tty:
            # Show spinner in TTY mode
            sys.stdout.write(f"{self.message}... ")
            sys.stdout.flush()
            self.thread = threading.Thread(target=self._spinner_loop, daemon=True)
            self.thread.start()
        else:
            # Just print the message in non-TTY mode
            sys.stdout.write(f"{self.message}... ")
            sys.stdout.flush()

    def _spinner_loop(self):
        """Run the spinner animation."""
        while self.running:
            frame = self.SPINNER_FRAMES[self._frame_index]
            sys.stdout.write(frame)
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write("\b")
            self._frame_index = (self._frame_index + 1) % len(self.SPINNER_FRAMES)

    def stop(self, success: bool = True, final_message: str | None = None):
        """
        Stop the progress indicator.

        Args:
            success: Whether the operation succeeded
            final_message: Optional message to display instead of success/fail indicator
        """
        if not self.running:
            return

        self.running = False

        if self.thread:
            self.thread.join(timeout=0.5)

        if self.is_tty:
            # Clear spinner
            sys.stdout.write("\b")

        # Show result
        if final_message:
            sys.stdout.write(f"{final_message}\n")
        elif success:
            sys.stdout.write("✓\n")
        else:
            sys.stdout.write("✗\n")

        sys.stdout.flush()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        success = exc_type is None
        self.stop(success=success)
        return False  # Don't suppress exceptions


@contextmanager
def progress(message: str, use_spinner: bool = None):
    """
    Context manager for progress indication.

    Args:
        message: Message to display during operation
        use_spinner: Force spinner on/off (auto-detected if None)

    Example:
        with progress("Fetching data"):
            data = fetch_data()
    """
    indicator = ProgressIndicator(message, use_spinner=use_spinner)
    indicator.start()
    try:
        yield indicator
    except BaseException:
        indicator.stop(success=False)
        raise
    else:
        indicator.stop(success=True)


class StepProgress:
    """Track progress through multiple steps."""

    def __init__(self, total_steps: int):
        """
        Initialize step progress tracker.

        Args:
            total_steps: Total number of steps
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.is_tty = sys.stdout.isatty()

    def step(self, message: str):
        """
        Move to the next step.

        Args:
            message: Description of current step

        Returns:
            Context manager for the step
        """
        self.current_step += 1
        step_info = f"({self.current_step}/{self.total_steps})"

        if self.is_tty:
            # Show inline progress for TTY
            full_message = f"{message} {step_info}"
        else:
            # Show simpler message for non-TTY
            full_message = f"[{step_info}] {message}"

        return progress(full_message)

    def finish(self, message: str = "All steps completed"):
        """
        Mark all steps as finished.

        Args:
            message: Completion message
        """
        if self.is_tty:
            sys.stdout.write(f"\n{message} ✓\n")
        else:
            sys.stdout.write(f"{message}\n")
        sys.stdout.flush()


@contextmanager
def step_progress(total_steps: int):
    """
    Context manager for step-based progress.

    Args:
        total_steps: Total number of steps

    Example:
        with step_progress(3) as sp:
            with sp.step("Fetching data"):
                fetch_data()
            with sp.step("Analyzing"):
                analyze_data()
            with sp.step("Generating report"):
                generate_report()
    """
    sp = StepProgress(total_steps)
    try:
        yield sp
    finally:
        pass  # StepProgress doesn't need cleanup


class DotProgress:
    """Simple dot-based progress for non-critical operations."""

    def __init__(self, message: str, dot_interval: float = 1.0):
        """
        Initialize dot progress.

        Args:
            message: Message to display
            dot_interval: Seconds between dots
        """
        self.message = message
        self.dot_interval = dot_interval
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self):
        """Start the dot progress."""
        if self.running:
            return

        sys.stdout.write(f"{self.message}")
        sys.stdout.flush()

        self.running = True
        self.thread = threading.Thread(target=self._dot_loop, daemon=True)
        self.thread.start()

    def _dot_loop(self):
        """Print dots periodically."""
        while self.running:
            time.sleep(self.dot_interval)
            if self.running:
                sys.stdout.write(".")
                sys.stdout.flush()

    def stop(self):
        """Stop the dot progress."""
        if not self.running:
            return

        self.running = False

        if self.thread:
            self.thread.join(timeout=self.dot_interval + 0.5)

        sys.stdout.write(" done\n")
        sys.stdout.flush()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
