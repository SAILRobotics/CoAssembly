"""Unit tests for structured proactive-agent decisions and mock tools."""

from __future__ import annotations

import unittest

from agent_tools import PlaceholderAgentTools, parse_decision, validate_decision


class AgentToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = {
            "active_parts": ["GEAR_ROD_ROW2"],
            "known_steps": ["r2_gear_rod"],
            "ready_steps": ["r2_gear_rod"],
            "step_inputs": {
                "r2_gear_rod": ["GEAR_ROD_ROW2"],
            },
            "trigger": "typed_user_request",
        }

    def test_valid_active_part_action(self) -> None:
        decision = parse_decision(
            '{"action":"approach_part",'
            '"arguments":{"part":"GEAR_ROD_ROW2","step_id":"r2_gear_rod"},'
            '"reason":"The user requested the red gear rod",'
            '"requires_confirmation":true}'
        )
        valid, problem = validate_decision(decision, self.context)
        self.assertTrue(valid, problem)
        self.assertIn(
            "GEAR_ROD_ROW2",
            PlaceholderAgentTools().execute(decision),
        )

    def test_rejects_inactive_part(self) -> None:
        decision = parse_decision(
            '{"action":"offer_part","arguments":{"part":"MISSING_PART"},'
            '"reason":"requested","requires_confirmation":true}'
        )
        valid, problem = validate_decision(decision, self.context)
        self.assertFalse(valid)
        self.assertIn("not active", problem)

    def test_rejects_blocked_step(self) -> None:
        decision = parse_decision(
            '{"action":"prepare_tool",'
            '"arguments":{"tool":"screwdriver","step_id":"blocked_step"},'
            '"reason":"prepare","requires_confirmation":false}'
        )
        context = {
            **self.context,
            "known_steps": [*self.context["known_steps"], "blocked_step"],
        }
        valid, problem = validate_decision(decision, context)
        self.assertFalse(valid)
        self.assertIn("not READY", problem)

    def test_motion_requires_confirmation(self) -> None:
        decision = parse_decision(
            '{"action":"approach_human",'
            '"arguments":{"reason":"offer fastening help"},'
            '"reason":"stage 4 is ready","requires_confirmation":false}'
        )
        valid, problem = validate_decision(decision, self.context)
        self.assertFalse(valid)
        self.assertIn("must require confirmation", problem)

    def test_proactive_action_requires_relevant_part(self) -> None:
        decision = parse_decision(
            '{"action":"offer_part",'
            '"arguments":{"part":"GEAR_ROD_ROW2"},'
            '"reason":"proactive offer","requires_confirmation":true}'
        )
        context = {
            **self.context,
            "trigger": "step_selected",
            "active_parts": ["GEAR_ROD_ROW2", "SCREW_ROW1_LEFT"],
            "selected_step": "r1_fasten_first_stand",
            "selected_step_details": {"state": "ready"},
            "known_steps": [
                *self.context["known_steps"],
                "r1_fasten_first_stand",
            ],
            "ready_steps": ["r1_fasten_first_stand"],
            "step_inputs": {
                **self.context["step_inputs"],
                "r1_fasten_first_stand": ["SCREW_ROW1_LEFT"],
            },
        }
        valid, problem = validate_decision(decision, context)
        self.assertFalse(valid)
        self.assertIn("not relevant", problem)

    def test_proactive_action_accepts_selected_step_input(self) -> None:
        decision = parse_decision(
            '{"action":"offer_part",'
            '"arguments":{"part":"SCREW_ROW1_LEFT"},'
            '"reason":"needed now","requires_confirmation":true}'
        )
        context = {
            **self.context,
            "trigger": "step_selected",
            "active_parts": ["SCREW_ROW1_LEFT"],
            "selected_step": "r1_fasten_first_stand",
            "selected_step_details": {"state": "ready"},
            "known_steps": [
                *self.context["known_steps"],
                "r1_fasten_first_stand",
            ],
            "ready_steps": ["r1_fasten_first_stand"],
            "step_inputs": {
                **self.context["step_inputs"],
                "r1_fasten_first_stand": ["SCREW_ROW1_LEFT"],
            },
        }
        valid, problem = validate_decision(decision, context)
        self.assertTrue(valid, problem)

    def test_omitted_motion_confirmation_defaults_to_true(self) -> None:
        decision = parse_decision(
            '{"action":"offer_part",'
            '"arguments":{"part":"GEAR_ROD_ROW2"},'
            '"reason":"The user requested the red gear rod"}'
        )
        self.assertTrue(decision.requires_confirmation)
        valid, problem = validate_decision(decision, self.context)
        self.assertTrue(valid, problem)


if __name__ == "__main__":
    unittest.main()
