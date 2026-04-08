import asyncio
import json
import os
import sys
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError as OpenAIRateLimitError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL", "https://router.huggingface.co/v1"
)
MODEL_NAME = os.getenv(
    "MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct"
)
# Nemotron-3-Super is used by judges in Phase 2 — this default
# covers the HF Router free tier for your own baseline run
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required.")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "customs-trade-classification"
MAX_STEPS = 30
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.60

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """You are an autonomous customs classification agent with access to a live HTS database lookup tool.

Your goal is to classify a real shipment step-by-step to maximize your reward score.

WORKFLOW — follow this exact order:
1. Call lookup_hs with the most likely 2-digit chapter (e.g. "39" for plastics, "85" for electrical).
2. Read the returned entries carefully. Call classify_chapter with the correct 2-digit chapter.
3. Call lookup_hs again with the 4-digit heading prefix (e.g. "3903") to see subheading options.
4. Call classify_heading with the correct 4-digit heading.
5. Call lookup_hs with the full heading to see all subheadings and their duty rates.
6. Call classify_subheading with the full 10-digit subheading (e.g. "3903.20.00.00").
7. Call check_duty with the exact duty rate string from the lookup results (e.g. "Free" or "3.5%").
8. Call lookup_sanctions with the country of origin name.
9. Call check_sanctions with "flagged" if sanctioned, "clear" if not.
10. Call submit with "hold" if flagged, "approve" if clear.

CRITICAL RULES:
- You MUST only use action_type values from the available_actions list in the observation.
- If "submit" appears in your available_actions, you MUST choose "submit" immediately. Do not do any more lookups.
- NEVER repeat a classification step you have already completed. Move forward in the workflow.
- NEVER guess subheadings from memory. Always use lookup_hs first.
- You MUST respond with ONLY a raw JSON object. No markdown, no backticks, no explanation.
- If classify_subheading returns reward 0.00 twice in a row with the same value, stop retrying. Use lookup_hs with the heading prefix again to find a different subheading option.
- You may attempt classify_subheading a maximum of 3 times total. After 3 attempts, call check_duty with your best duty rate guess and move forward.
- DO NOT guess the heading immediately. You MUST use lookup_hs with the 2-digit chapter to scan the options. If the correct item isn't in the first results, use lookup_hs with different 4-digit prefixes until you find a strong semantic match to the shipment description.

JSON FORMAT:
{"action_type": "...", "value": "...", "reasoning": "..."}

IDENTIFYING THE CHAPTER:
- Read the product description carefully.
- Use lookup_hs with your best 2-digit estimate based on the material or function.
- Read the returned entries. If nothing matches, try a different 2-digit prefix.
- Do not guess the chapter without looking it up first.

EXAMPLE PERFECT TRAJECTORY:
{"action_type": "lookup_hs", "value": "39", "reasoning": "Searching plastics chapter."}
{"action_type": "classify_chapter", "value": "39", "reasoning": "Found plastics chapter."}
{"action_type": "lookup_hs", "value": "3903", "reasoning": "Searching polymers of styrene."}
{"action_type": "classify_heading", "value": "3903", "reasoning": "Matches polymers of styrene."}
{"action_type": "lookup_hs", "value": "3903.20.00.00", "reasoning": "Searching subheading."}
{"action_type": "classify_subheading", "value": "3903.20.00.00", "reasoning": "Matches our specific polymer."}
{"action_type": "check_duty", "value": "Free", "reasoning": "Duty rate is Free."}
{"action_type": "lookup_sanctions", "value": "Japan", "reasoning": "Checking Japan."}
{"action_type": "check_sanctions", "value": "clear", "reasoning": "Japan is not sanctioned."}
{"action_type": "submit", "value": "approve", "reasoning": "Clear to approve."}"""

# ---------------------------------------------------------------------------
# Logging helpers — exact format required by judges
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=60, max=120),
    retry=retry_if_exception_type(OpenAIRateLimitError),
    reraise=True,
)
def get_model_action(
    client: OpenAI,
    observation: dict,
    history: List[str],
) -> dict:
    obs_text = (
        f"SHIPMENT:\n{observation.get('shipment_description', '')}\n\n"
        f"FEEDBACK: {observation.get('feedback', '')}\n"
        f"LOOKUP RESULTS: {observation.get('lookup_results', '') or 'None'}\n"
        f"AVAILABLE ACTIONS: {observation.get('available_actions', [])}\n"
        f"CURRENT SCORE: {observation.get('current_score', 0.0)}\n"
        f"STEPS REMAINING: {observation.get('step_budget_remaining', 0)}\n\n"
        f"HISTORY:\n" + ("\n".join(history[-20:]) if history else "None")
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs_text},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

async def run_task(
    client: OpenAI,
    task_id: str,
) -> tuple[float, bool, int, List[float]]:
    """Run one task episode. Returns (score, success, steps, rewards)."""

    try:
        from client import CustomsEnv
        from models import CustomsAction
    except ImportError:
        print(
            "[DEBUG] Could not import CustomsEnv — ensure server is running.",
            flush=True,
        )
        return 0.0, False, 0, []

    if LOCAL_IMAGE_NAME:
        env = await CustomsEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        # Fallback to local server connection if no image is specified
        env = CustomsEnv(base_url="http://127.0.0.1:7860")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    max_steps = {"task_easy": 10, "task_medium": 20, "task_hard": 30}[task_id]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        try:
            result = await env.reset(task_id=task_id)
            obs = result.observation
        except Exception as e:
            print(f"[DEBUG] Failed to reset env for task {task_id}: {e}", flush=True)
            # Log step 0 error, then gracefully fail this task
            log_step(step=0, action="reset", reward=0.0, done=True, error=str(e)[:80])
            score = 0.0
            return score, False, 0, []

        parse_error_count = 0
        for step in range(1, max_steps + 1):
            if result.done:
                break

            obs_dict = {
                "shipment_description": obs.shipment_description,
                "feedback": obs.feedback,
                "lookup_results": obs.lookup_results,
                "available_actions": obs.available_actions,
                "current_score": obs.current_score,
                "step_budget_remaining": obs.step_budget_remaining,
            }

            error_msg = None
            try:
                action_data = await asyncio.to_thread(get_model_action, client, obs_dict, history)
                action = CustomsAction(
                    action_type=action_data["action_type"],
                    value=action_data.get("value", ""),
                    reasoning=action_data.get("reasoning", ""),
                )
                action_str = f"{action.action_type}={action.value}"
            except Exception as exc:
                print(f"[DEBUG] Model request failed: {exc}", flush=True)
                error_msg = str(exc)[:80]
                parse_error_count += 1
                if parse_error_count >= 3:
                    # Circuit breaker: 3 consecutive JSON failures — cut losses
                    print(
                        f"[DEBUG] parse_error_count={parse_error_count} — "
                        "forcing submit=hold to salvage partial score.",
                        flush=True,
                    )
                    action = CustomsAction(
                        action_type="submit",
                        value="hold",
                        reasoning="circuit breaker: repeated parse failures",
                    )
                    action_str = "submit=hold(circuit-breaker)"
                else:
                    # Fallback: attempt a lookup if stuck
                    available = obs.available_actions
                    fallback_type = available[0] if available else "submit"
                    action = CustomsAction(
                        action_type=fallback_type,
                        value="39",
                        reasoning="fallback after parse error",
                    )
                    action_str = f"{fallback_type}=fallback"

            try:
                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
            except Exception as e:
                print(f"[DEBUG] Failed to execute step in env: {e}", flush=True)
                error_msg = str(e)[:80]
                result = None
                reward = 0.0
                done = True

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

            history.append(
                f"Step {step}: {action_str} -> reward {reward:+.2f} | "
                f"feedback: {obs.feedback[:80]}"
            )

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score, success, steps_taken, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    results = {}
    for task_id in TASKS:
        score, success, steps, rewards = await run_task(client, task_id)
        results[task_id] = {
            "score": score,
            "success": success,
            "steps": steps,
        }

    print("\n=== FINAL RESULTS ===", flush=True)
    for task_id, r in results.items():
        print(
            f"Task: {task_id} | Score: {r['score']:.4f} | "
            f"Success: {r['success']} | Steps: {r['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
