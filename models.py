from typing import List, Optional, Literal
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class CustomsAction(Action):
    action_type: Literal[
        "lookup_hs",
        "lookup_sanctions",
        "classify_chapter",
        "classify_heading",
        "classify_subheading",
        "check_duty",
        "check_sanctions",
        "submit",
    ]
    value: str = Field(..., description=(
        "lookup_hs: HS prefix to search e.g. '39' or '3903'. "
        "lookup_sanctions: country name to check e.g. 'Iran'. "
        "classify_chapter: 2-digit chapter e.g. '39'. "
        "classify_heading: 4-digit heading e.g. '3903'. "
        "classify_subheading: full subheading e.g. '3903.20.00.00'. "
        "check_duty: duty rate string e.g. 'Free' or '3.5%'. "
        "check_sanctions: 'flagged' or 'clear'. "
        "submit: 'approve' or 'hold'."
    ))
    reasoning: str = Field(default="", description="Agent's reasoning for this action.")


class CustomsObservation(Observation):
    shipment_description: str = Field(default="", description="Full shipment details.")
    feedback: str = Field(default="", description="Result of the last action.")
    available_actions: List[str] = Field(default_factory=list)
    task_brief: str = Field(default="")
    step_budget_remaining: int = Field(default=30)
    current_score: float = Field(default=0.0)
    lookup_results: str = Field(default="", description="Results from lookup actions.")


class CustomsState(State):
    product_description: str = ""
    country_of_origin: str = ""
    declared_value: float = 0.0
    importer_name: str = ""
    correct_chapter: str = ""
    correct_heading: str = ""
    correct_subheading: str = ""
    correct_duty_rate: str = ""
    sanctions_hit: bool = False
    correct_verdict: str = ""
    chapter_score: float = 0.0
    heading_score: float = 0.0
    subheading_score: float = 0.0
    duty_score: float = 0.0
    sanctions_score: float = 0.0
    verdict_score: float = 0.0
    subheading_attempts: int = 0
    consecutive_lookup_count: int = 0
    last_lookup_prefix: str = ""
    same_prefix_count: int = 0
    chapter_attempts: int = 0
    heading_attempts: int = 0
    duty_attempts: int = 0
    sanctions_attempts: int = 0
    lookup_hs_count: int = 0
    lookup_sanctions_count: int = 0
    seed: int = 0
    task_id: str = "task_easy"
    max_steps: int = 10
    current_step: int = 0
