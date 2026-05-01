"""
Extended MCP client that exposes prompts and resources alongside tools.

smolagents' MCPClient only surfaces tools. The underlying mcp.ClientSession
already supports the full protocol — we just need sync wrappers around the
async session methods, reusing mcpadapt's event loop and thread.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

from smolagents import MCPClient, Tool

from mcp.types import (
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ReadResourceResult,
)


class ExtendedMCPClient(MCPClient):
    """
    Adds prompt and resource access to smolagents' tools-only MCPClient.

    Reuses the live ClientSession(s) that mcpadapt already manages. All calls
    are bridged from sync to async via the adapter's background event loop.
    """

    def __init__(
        self,
        server_parameters,
        adapter_kwargs: dict[str, Any] | None = None,
        structured_output: bool = False,
        timeout: float = 30,
    ):
        super().__init__(
            server_parameters=server_parameters,
            adapter_kwargs=adapter_kwargs,
            structured_output=structured_output,
        )
        self._timeout = timeout


    def _run_sync(self, coro):
        """Run an async coroutine on mcpadapt's loop and block for the result."""
        return asyncio.run_coroutine_threadsafe(
            coro, self._adapter.loop
        ).result(timeout=self._timeout)

    @property
    def _sessions(self):
        """Crashes loudly if the adapter hasn't connected yet."""
        sessions = self._adapter.sessions
        if not sessions:
            raise RuntimeError("No active MCP sessions; call connect() first")
        return sessions
        

    def list_prompts(self, server_index: int = 0) -> ListPromptsResult:
        return self._run_sync(self._sessions[server_index].list_prompts())

    def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None, server_index: int = 0
    ) -> GetPromptResult:
        return self._run_sync(
            self._sessions[server_index].get_prompt(name, arguments)
        )


    def list_resources(self, server_index: int = 0) -> ListResourcesResult:
        return self._run_sync(self._sessions[server_index].list_resources())

    def list_resource_templates(self, server_index: int = 0) -> ListResourceTemplatesResult:
        return self._run_sync(
            self._sessions[server_index].list_resource_templates()
        )

    def read_resource(self, uri: str, server_index: int = 0) -> ReadResourceResult:
        return self._run_sync(self._sessions[server_index].read_resource(uri))


    def list_all_prompts(self) -> list[ListPromptsResult]:
        """Query every connected server for prompts."""
        return [self.list_prompts(i) for i in range(len(self._sessions))]

    def list_all_resources(self) -> list[ListResourcesResult]:
        """Query every connected server for resources."""
        return [self.list_resources(i) for i in range(len(self._sessions))]

    @property
    def server_count(self) -> int:
        return len(self._sessions)
