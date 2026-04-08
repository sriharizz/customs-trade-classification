from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import CustomsAction, CustomsObservation, CustomsState
except ImportError:
    from models import CustomsAction, CustomsObservation, CustomsState  # type: ignore


class CustomsEnv(EnvClient[CustomsAction, CustomsObservation, CustomsState]):

    def _step_payload(self, action: CustomsAction) -> dict:
        return {
            "action_type": action.action_type,
            "value": action.value,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs = payload.get("observation", {})
        return StepResult(
            observation=CustomsObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                shipment_description=obs.get("shipment_description", ""),
                feedback=obs.get("feedback", ""),
                available_actions=obs.get("available_actions", []),
                task_brief=obs.get("task_brief", ""),
                step_budget_remaining=obs.get("step_budget_remaining", 0),
                current_score=obs.get("current_score", 0.0),
                lookup_results=obs.get("lookup_results", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> CustomsState:
        return CustomsState(
            # ── BaseState fields ──────────────────────────────────────────
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            # ── Shipment identity ─────────────────────────────────────────
            product_description=payload.get("product_description", ""),
            country_of_origin=payload.get("country_of_origin", ""),
            declared_value=payload.get("declared_value", 0.0),
            importer_name=payload.get("importer_name", ""),
            # ── Ground-truth classification ───────────────────────────────
            correct_chapter=payload.get("correct_chapter", ""),
            correct_heading=payload.get("correct_heading", ""),
            correct_subheading=payload.get("correct_subheading", ""),
            correct_duty_rate=payload.get("correct_duty_rate", ""),
            sanctions_hit=payload.get("sanctions_hit", False),
            correct_verdict=payload.get("correct_verdict", ""),
            # ── Running scores ────────────────────────────────────────────
            chapter_score=payload.get("chapter_score", 0.0),
            heading_score=payload.get("heading_score", 0.0),
            subheading_score=payload.get("subheading_score", 0.0),
            duty_score=payload.get("duty_score", 0.0),
            sanctions_score=payload.get("sanctions_score", 0.0),
            verdict_score=payload.get("verdict_score", 0.0),
            # ── Attempt counters ──────────────────────────────────────────
            subheading_attempts=payload.get("subheading_attempts", 0),
            chapter_attempts=payload.get("chapter_attempts", 0),
            heading_attempts=payload.get("heading_attempts", 0),
            duty_attempts=payload.get("duty_attempts", 0),
            sanctions_attempts=payload.get("sanctions_attempts", 0),
            # ── Lookup-discipline counters ────────────────────────────────
            consecutive_lookup_count=payload.get("consecutive_lookup_count", 0),
            last_lookup_prefix=payload.get("last_lookup_prefix", ""),
            same_prefix_count=payload.get("same_prefix_count", 0),
            lookup_hs_count=payload.get("lookup_hs_count", 0),
            lookup_sanctions_count=payload.get("lookup_sanctions_count", 0),
            # ── Episode metadata ──────────────────────────────────────────
            seed=payload.get("seed", 0),
            task_id=payload.get("task_id", "task_easy"),
            max_steps=payload.get("max_steps", 10),
            current_step=payload.get("current_step", 0),
        )
