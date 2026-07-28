"""Validated placeholder tools for proactive VLM-driven robot assistance.

These tools deliberately do not control hardware. They make the VLM's proposed
semantic action visible so its judgment can be evaluated before a safe robot
execution adapter is connected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


ACTION_DESCRIPTIONS = {
    "announce": "Say a short message to the user.",
    "approach_human": "Move to a predefined safe human-assistance pose.",
    "approach_part": "Move near a named active part without grasping it.",
    "offer_part": "Fetch or present a named active part to the user.",
    "prepare_tool": "Prepare a named tool for a task step.",
    "request_confirmation": "Ask the user to approve a proposed robot action.",
    "no_action": "Take no action because intervention is unnecessary or unsafe.",
}

PART_ACTIONS = {"approach_part", "offer_part"}
MOTION_ACTIONS = {"approach_human", "approach_part", "offer_part"}
EXPLICIT_USER_TRIGGERS = {"user_request", "typed_user_request"}


@dataclass
class AgentDecision:
    """One structured semantic action proposed by the VLM."""

    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    requires_confirmation: bool = False


def decision_schema_text() -> str:
    """Return the tool catalog and required JSON response format for the VLM."""
    tools = "\n".join(
        f'- "{name}": {description}'
        for name, description in ACTION_DESCRIPTIONS.items()
    )
    return (
        "Available actions:\n"
        f"{tools}\n\n"
        "Return exactly one JSON object with this shape:\n"
        "{\n"
        '  "action": "<available action name>",\n'
        '  "arguments": {"part": "...", "tool": "...", "step_id": "...", '
        '"message": "...", "reason": "..."},\n'
        '  "reason": "<why this action is useful now>",\n'
        '  "requires_confirmation": true\n'
        "}\n"
        "Include only argument fields needed by the chosen action. "
        "Use no Markdown and no text outside the JSON object."
    )


def parse_decision(raw: str) -> AgentDecision:
    """Parse a JSON decision, tolerating an accidental fenced code block."""
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("agent response must be a JSON object")
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("agent decision arguments must be an object")
    action = str(data.get("action", "")).strip()

    # If the model omits the confirmation field, fail safely by requiring
    # confirmation for motion. An explicit false is preserved and rejected by
    # validate_decision(), so the model cannot opt out of the safety gate.
    requires_confirmation = bool(data.get(
        "requires_confirmation",
        action in MOTION_ACTIONS,
    ))
    return AgentDecision(
        action=action,
        arguments=arguments,
        reason=str(data.get("reason", "")).strip(),
        requires_confirmation=requires_confirmation,
    )


def validate_decision(
    decision: AgentDecision,
    context: dict[str, Any],
) -> tuple[bool, str]:
    """Validate model output against allowed tools and current task state."""
    if decision.action not in ACTION_DESCRIPTIONS:
        return False, f"unknown action {decision.action!r}"

    known_steps = set(context.get("known_steps", []))
    ready_steps = set(context.get("ready_steps", []))
    active_parts = set(context.get("active_parts", []))

    step_id = str(decision.arguments.get("step_id", "")).strip()
    if step_id and step_id not in known_steps:
        return False, f"unknown step {step_id!r}"
    if step_id and decision.action != "no_action" and step_id not in ready_steps:
        return False, f"step {step_id!r} is not READY"

    if decision.action in PART_ACTIONS:
        part = str(decision.arguments.get("part", "")).strip()
        if not part:
            return False, f"{decision.action} requires a part"
        if part not in active_parts:
            return False, f"part {part!r} is not active"

        # A direct user request may name any currently active part. A proactive
        # decision must instead be relevant to the step that caused or frames
        # the decision, preventing unsolicited offers of unrelated components.
        trigger = str(context.get("trigger", ""))
        if trigger not in EXPLICIT_USER_TRIGGERS:
            step_inputs = context.get("step_inputs", {})
            relevant_step_ids: list[str] = []

            if step_id:
                relevant_step_ids.append(step_id)
            else:
                selected = context.get("selected_step")
                selected_details = context.get("selected_step_details", {})
                if selected and selected_details.get("state") == "ready":
                    relevant_step_ids.append(str(selected))
                relevant_step_ids.extend(context.get("controller_open_steps", []))
                if not relevant_step_ids:
                    relevant_step_ids.extend(context.get("ready_steps", []))

            relevant_parts = {
                input_part
                for relevant_step_id in relevant_step_ids
                for input_part in step_inputs.get(relevant_step_id, [])
            }
            if part not in relevant_parts:
                scope = ", ".join(dict.fromkeys(relevant_step_ids)) or "READY steps"
                return False, (
                    f"part {part!r} is not relevant to the current step scope: {scope}"
                )

    if decision.action == "announce" and not decision.arguments.get("message"):
        return False, "announce requires a message"
    if decision.action == "approach_human" and not decision.arguments.get("reason"):
        return False, "approach_human requires an argument named reason"
    if decision.action == "prepare_tool" and not decision.arguments.get("tool"):
        return False, "prepare_tool requires a tool"
    if decision.action == "request_confirmation" and not decision.arguments.get("reason"):
        return False, "request_confirmation requires an argument named reason"
    if decision.action == "no_action" and not (
        decision.reason or decision.arguments.get("reason")
    ):
        return False, "no_action requires a reason"

    # Motion is mock-only today, but keep confirmation explicit in the decision
    # so the same interface can be safely gated when hardware is connected.
    if decision.action in MOTION_ACTIONS and not decision.requires_confirmation:
        return False, f"{decision.action} must require confirmation"

    return True, ""


class PlaceholderAgentTools:
    """Execute validated semantic tools as visible mock actions only."""

    def execute(self, decision: AgentDecision) -> str:
        action = decision.action
        args = decision.arguments

        if action == "announce":
            result = f'Robot would say: "{args["message"]}"'
        elif action == "approach_human":
            result = f'Robot would approach the safe assistance pose: {args["reason"]}'
        elif action == "approach_part":
            result = f'Robot would approach active part {args["part"]}'
        elif action == "offer_part":
            result = f'Robot would fetch or present active part {args["part"]}'
        elif action == "prepare_tool":
            step = f' for {args["step_id"]}' if args.get("step_id") else ""
            result = f'Robot would prepare tool {args["tool"]}{step}'
        elif action == "request_confirmation":
            result = f'Robot would ask for confirmation: {args["reason"]}'
        else:
            result = f'Robot would take no action: {decision.reason or args["reason"]}'

        if decision.requires_confirmation:
            result += " [confirmation required; mock action not executed]"
        else:
            result += " [mock action only]"
        print(f"[ProactiveAgent] {result}")
        return result
