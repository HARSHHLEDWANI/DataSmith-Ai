from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def restore_registry():
    from app.agent.session_store import conversation_store
    from app.tools.registry import register_all_tools

    reg = register_all_tools()
    saved = {name: reg.get(name).func for name in reg.names()}
    conversation_store.reset()
    yield
    for name, func in saved.items():
        tool = reg.get(name)
        if tool is not None:
            tool.func = func
