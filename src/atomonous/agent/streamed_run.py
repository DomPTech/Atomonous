from dataclasses import dataclass
from typing import Callable, Iterator, Optional
from smolagents import FinalAnswerStep

@dataclass
class StreamedRun:
    """
    Convenient wrapper around a smolagents run for streaming output.
    """
    _it_factory: Callable[[], Iterator]
    _it: Optional[Iterator] = None
    _final: Optional[FinalAnswerStep] = None
    _done: bool = False

    def stream(self) -> Iterator:
        if self._done:
            return iter(())
        if self._it is None:
            self._it = self._it_factory()

        for item in self._it:
            if isinstance(item, FinalAnswerStep):
                self._final = item
            yield item

        self._done = True

    def final(self) -> FinalAnswerStep:
        if not self._done:
            for _ in self.stream():
                pass
        return self._final