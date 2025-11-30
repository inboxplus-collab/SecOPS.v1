# backend/src/extensions/__init__.py
"""
SecOps AI – extensions package.

This package contains optional, advanced modules for:
- adaptive / auto-evolving behavior,
- simple machine learning style optimization (multi-armed bandits),
- feature storage for signals coming from the platform,
- policy tuning based on feedback.

These modules are completely optional and are not imported by the core
backend. You can safely ignore them until you are ready to integrate.

Included integrations:
- `agent0_adapter`: keeps data structures compatible with the Agent0 project
  (https://github.com/aiming-lab/Agent0) so you can later introduce their
  runtime without refactoring business logic.
"""
