import json
import os
import random
import uuid
import sys
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..models import CustomsAction, CustomsObservation, CustomsState
except ImportError:
    from models import CustomsAction, CustomsObservation, CustomsState  # type: ignore

from openenv.core.env_server import Environment

# ---------------------------------------------------------------------------
# Reward weights — must sum to 1.0
# ---------------------------------------------------------------------------

REWARD_WEIGHTS = {
    "chapter":    0.15,
    "heading":    0.20,
    "subheading": 0.25,
    "duty":       0.20,
    "sanctions":  0.10,
    "verdict":    0.10,
}

TASK_CONFIG = {
    "task_easy":   {"max_steps": 35},
    "task_medium": {"max_steps": 35},
    "task_hard":   {"max_steps": 35},
}

TASK_BRIEF = (
    "You are a customs classification agent. Use lookup_hs to search "
    "the HTS database, then classify the shipment step by step. "
    "Use lookup_sanctions to verify the country of origin. "
    "Submit a final verdict of approve or hold."
)

# ---------------------------------------------------------------------------
# Shipment catalogue
# ---------------------------------------------------------------------------

SHIPMENTS = [
    # ── EASY ────────────────────────────────────────────────────────────────
    # Very famous, unambiguous products. Agent gets 100% classification.
    {
        "product_description": (
            "Nuclear reactor vessel, pressurised water type, VVER-1200 design, "
            "rated thermal output 3,200 MW, forged low-alloy steel body with stainless cladding. "
            "Not for military use."
        ),
        "country_of_origin": "Russia",
        "declared_value": 890000.0,
        "importer_name": "Atom Energy Equipment India",
        "hs_chapter": "84",
        "hs_heading": "8401",
        "hs_subheading": "8401.10.00.00",
        "difficulty": "easy",
    },
    {
        "product_description": (
            "Plastic buckets and pails, injection-moulded, made of polypropylene, "
            "capacity 10 litres, with moulded-in handles."
        ),
        "country_of_origin": "Germany",
        "declared_value": 18000.0,
        "importer_name": "EuroPlas Distribution India Pvt Ltd",
        "hs_chapter": "39",
        "hs_heading": "3926",
        "hs_subheading": "3926.90.10.00",
        "difficulty": "easy",
    },
    {
        "product_description": (
            "Cast iron flywheels for industrial machinery, "
            "diameter 450 mm, bore 60 mm, weight 28 kg each."
        ),
        "country_of_origin": "South Korea",
        "declared_value": 35000.0,
        "importer_name": "Korean Machinery Parts India",
        "hs_chapter": "84",
        "hs_heading": "8483",
        "hs_subheading": "8483.50.60.00",
        "difficulty": "easy",
    },
    {
        "product_description": (
            "Plastic rainwear — ponchos and full-length waterproof coats — "
            "made of PVC-coated polyester fabric."
        ),
        "country_of_origin": "Japan",
        "declared_value": 22000.0,
        "importer_name": "Tokyo Rainwear Imports India",
        "hs_chapter": "39",
        "hs_heading": "3926",
        "hs_subheading": "3926.20.60.00",
        "difficulty": "easy",
    },
    {
        "product_description": (
            "Mass spectrometer, quadrupole type, for analytical laboratory use, "
            "mass range 1 to 1,024 amu, electron ionization source."
        ),
        "country_of_origin": "Finland",
        "declared_value": 320000.0,
        "importer_name": "Nordic Analytical India Pvt Ltd",
        "hs_chapter": "90",
        "hs_heading": "9027",
        "hs_subheading": "9027.81.00.00",
        "difficulty": "easy",
    },

    # ── MEDIUM ───────────────────────────────────────────────────────────────
    # Slightly ambiguous. The agent gets subheading partially right, but generally passes.
    {
        "product_description": (
            "Three-phase AC induction motor, rated output 5 kW, 50 Hz, 400V, "
            "output exceeds 750 W but does NOT exceed 14.92 kW, "
            "IE2 efficiency class, frame IEC 112M, foot-mounted."
        ),
        "country_of_origin": "China",
        "declared_value": 87000.0,
        "importer_name": "Sinotech Power Equipment India",
        "hs_chapter": "85",
        "hs_heading": "8501",
        "hs_subheading": "8501.52.40.00",
        "difficulty": "medium",
    },
    {
        "product_description": (
            "Hydraulic torque converter assembly for industrial power transmission, "
            "single-stage, three-element type, rated input 560 kW at 1,800 rpm."
        ),
        "country_of_origin": "Germany",
        "declared_value": 145000.0,
        "importer_name": "Voith India Power Transmissions",
        "hs_chapter": "84",
        "hs_heading": "8483",
        "hs_subheading": "8483.40.10.00",
        "difficulty": "medium",
    },
    {
        "product_description": (
            "Microwave oven industrial heating unit, 2.45 GHz magnetron array, "
            "continuous conveyor type, cavity dimensions 600 x 400 x 300 mm."
        ),
        "country_of_origin": "South Korea",
        "declared_value": 58000.0,
        "importer_name": "POSCO Catering Equipment India",
        "hs_chapter": "85",
        "hs_heading": "8514",
        "hs_subheading": "8514.20.40.00",
        "difficulty": "medium",
    },

    # ── HARD ─────────────────────────────────────────────────────────────────
    # Very ambiguous subheadings WITH sanctioned countries. Agent fails classification but gets sanctions.
    {
        "product_description": (
            "Centrifuge, general industrial type, basket diameter 1,200 mm, "
            "stainless steel bowl, variable speed drive 0 to 1,800 rpm."
        ),
        "country_of_origin": "Iran",
        "declared_value": 185000.0,
        "importer_name": "Tehran Industrial Equipment India",
        "hs_chapter": "84",
        "hs_heading": "8421",
        "hs_subheading": "8421.19.00.00",
        "difficulty": "hard",
    },
    {
        "product_description": (
            "Industrial spray dryer, rotary atomizer type, "
            "evaporative capacity 500 kg/h water, "
            "inlet air temperature 220 deg C, outlet 90 deg C."
        ),
        "country_of_origin": "Russia",
        "declared_value": 320000.0,
        "importer_name": "Moscow Food Equipment India",
        "hs_chapter": "84",
        "hs_heading": "8419",
        "hs_subheading": "8419.39.02",
        "difficulty": "hard",
    },
    {
        "product_description": (
            "Heat exchanger, shell and tube type, fixed tubesheet, "
            "shell diameter 600 mm, tube length 4,800 mm."
        ),
        "country_of_origin": "North Korea",
        "declared_value": 95000.0,
        "importer_name": "Pyongyang Process Equipment India",
        "hs_chapter": "84",
        "hs_heading": "8419",
        "hs_subheading": "8419.50.10.00",
        "difficulty": "hard",
    },
    {
        "product_description": (
            "Hydraulic press, four-column type, rated force 500 tonnes, "
            "daylight opening 1,200 mm, table size 1,000 x 800 mm."
        ),
        "country_of_origin": "Syria",
        "declared_value": 145000.0,
        "importer_name": "Damascus Machinery India",
        "hs_chapter": "84",
        "hs_heading": "8462",
        "hs_subheading": "8462.11.00",
        "difficulty": "hard",
    },
    {
        "product_description": (
            "Vacuum furnace, cold wall type, molybdenum hot zone, "
            "working volume 600 x 600 x 900 mm, maximum temperature 1,400 deg C."
        ),
        "country_of_origin": "Belarus",
        "declared_value": 380000.0,
        "importer_name": "Minsk Thermal Processing India",
        "hs_chapter": "85",
        "hs_heading": "8514",
        "hs_subheading": "8514.11.00.00",
        "difficulty": "hard",
    }
]

# --------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    hts_path = os.path.join(data_dir, "hts_data.json")
    with open(hts_path, "r", encoding="utf-8") as f:
        hts_raw = json.load(f)

    hts_lookup = {}
    for entry in hts_raw:
        key = entry["htsno"].strip()
        hts_lookup[key] = {
            "description": entry["description"],
            "duty_rate": entry["general"],
        }

    ofac_path = os.path.join(data_dir, "ofac_sdn.csv")
    sanctioned = set()
    with open(ofac_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        country = line.strip().strip('"')
        if country:
            sanctioned.add(country.upper())

    return hts_lookup, sanctioned


HTS_LOOKUP, SANCTIONED_COUNTRIES = load_data()

HARDCODED_SANCTIONED = {
    "NORTH KOREA", "DPRK", "IRAN", "RUSSIA", "SYRIA",
    "BELARUS", "CUBA", "VENEZUELA", "MYANMAR", "SUDAN",
    "ZIMBABWE", "LIBYA", "SOMALIA", "IRAQ", "LEBANON",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_duty_rate(subheading: str) -> str:
    if subheading in HTS_LOOKUP:
        return HTS_LOOKUP[subheading]["duty_rate"]
    prefix7 = subheading[:7]
    for key, val in HTS_LOOKUP.items():
        if key[:7] == prefix7:
            return val["duty_rate"]
    prefix4 = subheading[:4]
    for key, val in HTS_LOOKUP.items():
        if key[:4] == prefix4:
            return val["duty_rate"]
    return "Free"


def is_sanctioned(country: str) -> bool:
    cu = country.upper().strip()
    if cu in HARDCODED_SANCTIONED:
        return True
    if cu in SANCTIONED_COUNTRIES:
        return True
    for entry in SANCTIONED_COUNTRIES:
        for token in HARDCODED_SANCTIONED:
            if token in entry and cu in token:
                return True
    return False


def _hs_lookup_results(prefix: str, max_results: int = 20) -> str:
    """Return up to max_results HTS entries whose key starts with prefix."""
    prefix = prefix.strip().replace(".", "").replace(" ", "")
    matches = {}
    for key, val in HTS_LOOKUP.items():
        normalized = key.replace(".", "").replace(" ", "")
        if normalized.startswith(prefix):
            matches[key] = val
        if len(matches) >= max_results:
            break
    if not matches:
        return f"No HTS entries found for prefix '{prefix}'. Try a shorter prefix."
    lines = [f"{k}: {v['description']} | Duty: {v['duty_rate']}" for k, v in matches.items()]
    return "\n".join(lines)


def _grade_chapter(agent_val: str, correct: str) -> float:
    """Full credit for exact match. Half credit for same HS section."""
    if agent_val.strip() == correct.strip():
        return 1.0
    # Same first digit = same HS section, partial credit
    if agent_val.strip()[:1] == correct.strip()[:1]:
        return 0.4
    return 0.0


def _grade_heading(agent_val: str, correct_heading: str, correct_chapter: str) -> float:
    agent = agent_val.strip()
    if agent == correct_heading:
        return 1.0
    # First 2 digits match the chapter
    if agent[:2] == correct_chapter:
        return 0.4
    return 0.0


def _grade_subheading(agent_val: str, correct: str) -> float:
    """Full credit for exact match. Partial credit (0.4) if first 6 digits match."""
    if not correct.strip():
        return 0.0
    
    a = agent_val.strip().replace(".", "").replace(" ", "")
    c = correct.strip().replace(".", "").replace(" ", "")
    
    if a == c:
        return 1.0
    
    if len(a) >= 6 and len(c) >= 6 and a[:6] == c[:6]:
        return 0.4
        
    return 0.0


# Synonyms for a zero / duty-free rate
_FREE_SYNONYMS = {"free", "exempt", "none", "nil", "zero", "n/a", "na"}


def _grade_duty(agent_val: str, correct: str) -> float:
    """Grade a duty-rate answer with 'Free'-aware normalisation.

    Sanitise both strings first so that Free / 0 / exempt are treated
    as equivalent before any numeric proximity check is attempted.
    """
    a = agent_val.lower().strip().replace(" ", "").replace("%", "")
    c = correct.lower().strip().replace(" ", "").replace("%", "")
    # Normalise free-duty synonyms to the canonical string "0"
    if a in _FREE_SYNONYMS:
        a = "0"
    if c in _FREE_SYNONYMS:
        c = "0"
    if a == c:
        return 1.0
    if a in c or c in a:
        return 0.7
    # Numeric proximity check (both strings are now guaranteed non-synonym)
    try:
        af = float(a)
        cf = float(c)
        if abs(af - cf) <= 0.5:
            return 0.7
        if abs(af - cf) <= 2.0:
            return 0.3
    except ValueError:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomsEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = CustomsState()
        self._rng = random.Random()
        self._prev_score = 0.0

    def reset(
        self, seed=None, episode_id=None, task_id="task_easy", **kwargs
    ) -> CustomsObservation:

        seed = seed if seed is not None else random.randint(0, 99999)
        self._rng = random.Random(seed)
        self._prev_score = 0.0

        if task_id == "task_easy":
            candidates = [s for s in SHIPMENTS if s["difficulty"] == "easy"]
        elif task_id == "task_medium":
            candidates = [s for s in SHIPMENTS if s["difficulty"] == "medium"]
        else:
            candidates = [s for s in SHIPMENTS if s["difficulty"] == "hard"]

        if not candidates:
            candidates = SHIPMENTS

        shipment_index = seed % len(candidates)
        shipment = candidates[shipment_index]
        duty_rate = get_duty_rate(shipment["hs_subheading"])
        sanctioned = is_sanctioned(shipment["country_of_origin"])
        verdict = "hold" if sanctioned else "approve"
        max_steps = TASK_CONFIG[task_id]["max_steps"]

        self._state = CustomsState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_step=0,
            product_description=shipment["product_description"],
            country_of_origin=shipment["country_of_origin"],
            declared_value=shipment["declared_value"],
            importer_name=shipment["importer_name"],
            correct_chapter=shipment["hs_chapter"],
            correct_heading=shipment["hs_heading"],
            correct_subheading=shipment["hs_subheading"],
            correct_duty_rate=duty_rate,
            sanctions_hit=sanctioned,
            correct_verdict=verdict,
            chapter_score=0.0,
            heading_score=0.0,
            subheading_score=0.0,
            duty_score=0.0,
            sanctions_score=0.0,
            verdict_score=0.0,
            subheading_attempts=0,
            consecutive_lookup_count=0,
            last_lookup_prefix="",
            same_prefix_count=0,
            chapter_attempts=0,
            heading_attempts=0,
            lookup_hs_count=0,
            lookup_sanctions_count=0,
            seed=seed,
            task_id=task_id,
            max_steps=max_steps,
        )

        shipment_text = (
            f"Product: {shipment['product_description']}\n"
            f"Declared Value: USD {shipment['declared_value']:,.2f}\n"
            f"Importer: {shipment['importer_name']}"
        )

        return CustomsObservation(
            done=False,
            reward=None,
            shipment_description=shipment_text,
            feedback=(
                "New shipment received. Start by using lookup_hs with a 2-digit "
                "chapter prefix to search the HTS database."
            ),
            available_actions=[
                "lookup_hs",
                "lookup_sanctions",
                "classify_chapter",
            ],
            task_brief=TASK_BRIEF,
            step_budget_remaining=max_steps,
            current_score=0.0,
            lookup_results="",
        )

    def step(
        self, action: CustomsAction, timeout_s=None, **kwargs
    ) -> CustomsObservation:

        self._state.step_count += 1
        self._state.current_step += 1
        steps_used = self._state.current_step
        max_steps = self._state.max_steps
        steps_remaining = max(0, max_steps - steps_used)

        action_type = action.action_type.strip().lower()
        value = action.value.strip()

        feedback = ""
        lookup_results = ""
        reward = 0.0

        # ── LOOKUP ACTIONS (no reward, just information) ──────────────────

        if action_type == "lookup_hs":
            self._state.lookup_hs_count += 1
            self._state.consecutive_lookup_count += 1

            # Track repeated same prefix
            if value == self._state.last_lookup_prefix:
                self._state.same_prefix_count += 1
            else:
                self._state.same_prefix_count = 1
                self._state.last_lookup_prefix = value

            results = _hs_lookup_results(value)
            lookup_results = results

            # Force progress if agent keeps looking up same prefix 3+ times
            if self._state.same_prefix_count >= 3:
                feedback = (
                    f"You have searched '{value}' {self._state.same_prefix_count} times. "
                    f"The results are the same. You MUST now make a classification decision "
                    f"using the available_actions list. Stop looking up '{value}'."
                )
            else:
                lines = results.splitlines()
                feedback = f"HTS lookup for '{value}' returned {len(lines)} entries. Read them carefully."

            next_actions = self._get_available_actions(include_lookups=True)

        elif action_type == "lookup_sanctions":
            self._state.lookup_sanctions_count += 1
            country = value.strip()
            hit = is_sanctioned(country)
            if hit:
                lookup_results = (
                    f"OFAC CHECK: '{country}' appears on the US sanctions list. "
                    f"Shipments from this country must be HELD."
                )
            else:
                lookup_results = (
                    f"OFAC CHECK: '{country}' is NOT on the US sanctions list. "
                    f"No sanctions restriction applies."
                )
            feedback = f"Sanctions lookup completed for '{country}'."
            next_actions = self._get_available_actions(include_lookups=True)

        # ── CLASSIFICATION ACTIONS (graded, reward-bearing) ───────────────

        elif action_type == "classify_chapter":
            if self._state.chapter_score == 0.0:
                self._state.chapter_attempts += 1
                grade = _grade_chapter(value, self._state.correct_chapter)
                self._state.chapter_score = grade * REWARD_WEIGHTS["chapter"]
                if grade == 1.0:
                    feedback = f"Chapter {value} correct. Proceed to classify_heading."
                elif grade > 0:
                    feedback = (
                        f"Partially correct — same HS section. "
                        f"Correct chapter starts with '{self._state.correct_chapter[:1]}'. "
                        f"Use lookup_hs with a different 2-digit prefix."
                    )
                else:
                    if self._state.chapter_attempts >= 3:
                        # Force accept best guess and move on
                        self._state.chapter_score = 0.0
                        feedback = (
                            f"Still incorrect after {self._state.chapter_attempts} attempts. "
                            f"Moving forward with partial score."
                        )
                    else:
                        feedback = (
                            f"Incorrect chapter. Try lookup_hs with a different 2-digit prefix."
                        )
            else:
                feedback = "Chapter already classified."
            next_actions = self._get_available_actions(include_lookups=True)

        elif action_type == "classify_heading":
            if self._state.chapter_score == 0.0:
                feedback = "You must classify_chapter before classify_heading."
                next_actions = self._get_available_actions(include_lookups=True)
            elif self._state.heading_score == 0.0:
                self._state.heading_attempts += 1
                grade = _grade_heading(
                    value,
                    self._state.correct_heading,
                    self._state.correct_chapter,
                )
                self._state.heading_score = grade * REWARD_WEIGHTS["heading"]
                if grade == 1.0:
                    feedback = f"Heading {value} correct. Proceed to classify_subheading."
                elif grade > 0:
                    feedback = (
                        f"Chapter prefix matched but heading is wrong. "
                        f"Use lookup_hs with '{self._state.correct_chapter}' to narrow down."
                    )
                else:
                    if self._state.heading_attempts >= 3:
                        feedback = (
                            f"Still incorrect after {self._state.heading_attempts} attempts. "
                            f"Moving forward with partial score."
                        )
                    else:
                        feedback = (
                            f"Incorrect heading. Use lookup_hs with chapter "
                            f"'{self._state.correct_chapter}' to find the right heading."
                        )
                next_actions = self._get_available_actions(include_lookups=True)
            else:
                feedback = "Heading already classified."
                next_actions = self._get_available_actions(include_lookups=True)

        elif action_type == "classify_subheading":
            if self._state.heading_score == 0.0:
                feedback = "You must classify_heading before classify_subheading."
                next_actions = self._get_available_actions(include_lookups=True)
            else:
                self._state.subheading_attempts += 1
                grade = _grade_subheading(value, self._state.correct_subheading)
                heading_grade = self._state.heading_score / REWARD_WEIGHTS["heading"]
                if heading_grade < 1.0:
                    grade = 0.0
                new_score = grade * REWARD_WEIGHTS["subheading"]
                if new_score > self._state.subheading_score:
                    self._state.subheading_score = new_score
                if grade >= 0.99:
                    feedback = "Subheading confirmed. Proceed to check_duty."
                elif self._state.subheading_attempts >= 3:
                    feedback = "Maximum subheading attempts reached. Proceeding to check_duty."
                elif grade >= 0.5:
                    feedback = "Close — refine further using lookup_hs."
                else:
                    feedback = "Incorrect. Use lookup_hs with the heading prefix to find options."
                next_actions = self._get_available_actions(include_lookups=True)

        elif action_type == "check_duty":
            if self._state.subheading_attempts == 0:
                feedback = "You must classify_subheading before check_duty."
                next_actions = self._get_available_actions(include_lookups=True)
            elif self._state.duty_score == 0.0:
                self._state.duty_attempts += 1
                grade = _grade_duty(value, self._state.correct_duty_rate)
                self._state.duty_score = grade * REWARD_WEIGHTS["duty"]
                if grade == 1.0:
                    feedback = (
                        f"Duty rate '{self._state.correct_duty_rate}' confirmed. "
                        f"Proceed to check_sanctions."
                    )
                elif grade > 0:
                    feedback = (
                        f"Close but not exact. Correct rate is near your answer. "
                        f"Use lookup_hs on the subheading to find the exact general rate."
                    )
                else:
                    if self._state.duty_attempts >= 3:
                        feedback = (
                            f"Still incorrect after {self._state.duty_attempts} attempts. "
                            f"Moving forward with partial score."
                        )
                    else:
                        feedback = (
                            f"Incorrect duty rate. Use lookup_hs with the full subheading "
                            f"to find the general duty rate."
                        )
                next_actions = self._get_available_actions(include_lookups=True)
            else:
                feedback = "Duty already checked."
                next_actions = self._get_available_actions(include_lookups=True)

        elif action_type == "check_sanctions":
            if self._state.sanctions_score == 0.0:
                self._state.sanctions_attempts += 1
                av = value.lower().strip()
                if self._state.sanctions_hit:
                    if av in ["flagged", "sanctioned", "yes", "true", "hold"]:
                        self._state.sanctions_score = REWARD_WEIGHTS["sanctions"]
                        feedback = "Correct. Country is OFAC-sanctioned. Submit 'hold'."
                    else:
                        if self._state.sanctions_attempts >= 3:
                            feedback = (
                                f"Still incorrect after {self._state.sanctions_attempts} attempts. "
                                f"Moving forward with partial score."
                            )
                        else:
                            feedback = (
                                "Incorrect. Use lookup_sanctions with the country name "
                                "to verify OFAC status."
                            )
                else:
                    if av in ["clear", "clean", "no", "false", "approve", "not sanctioned"]:
                        self._state.sanctions_score = REWARD_WEIGHTS["sanctions"]
                        feedback = "Correct. Country is clear. Submit 'approve'."
                    else:
                        if self._state.sanctions_attempts >= 3:
                            feedback = (
                                f"Still incorrect after {self._state.sanctions_attempts} attempts. "
                                f"Moving forward with partial score."
                            )
                        else:
                            feedback = (
                                "Incorrect. Use lookup_sanctions with the country name "
                                "to verify OFAC status."
                            )
                next_actions = self._get_available_actions(include_lookups=True)
            else:
                feedback = "Sanctions already checked."
                next_actions = self._get_available_actions(include_lookups=True)

        elif action_type == "submit":
            if self._state.verdict_score == 0.0:
                av = value.lower().strip()
                if av == self._state.correct_verdict:
                    self._state.verdict_score = REWARD_WEIGHTS["verdict"]
                    feedback = (
                        f"Correct verdict: {self._state.correct_verdict}. "
                        f"Classification complete."
                    )
                else:
                    feedback = (
                        f"Incorrect verdict. Expected '{self._state.correct_verdict}'. "
                        f"Review your sanctions check before submitting."
                    )
            else:
                feedback = "Already submitted."
            next_actions = []

        else:
            feedback = f"Unknown action_type: '{action_type}'."
            next_actions = self._get_available_actions(include_lookups=True)

        # ── REWARD COMPUTATION ────────────────────────────────────────────

        total_score = (
            self._state.chapter_score
            + self._state.heading_score
            + self._state.subheading_score
            + self._state.duty_score
            + self._state.sanctions_score
            + self._state.verdict_score
        )


        reward = round(total_score - self._prev_score, 4)
        self._prev_score = total_score

        done = (action_type == "submit") or (steps_remaining <= 0)

        shipment_text = (
            f"Product: {self._state.product_description}\n"
            f"Declared Value: USD {self._state.declared_value:,.2f}\n"
            f"Importer: {self._state.importer_name}"
        )

        return CustomsObservation(
            done=done,
            reward=reward,
            shipment_description=shipment_text,
            feedback=feedback,
            available_actions=next_actions if not done else [],
            task_brief=TASK_BRIEF,
            step_budget_remaining=steps_remaining,
            current_score=round(total_score, 4),
            lookup_results=lookup_results,
        )

    @property
    def state(self) -> CustomsState:
        return self._state

    def _get_available_actions(self, include_lookups: bool = True) -> list:
        actions = []
        if include_lookups:
            actions += ["lookup_hs", "lookup_sanctions"]

        if self._state.chapter_score == 0.0 and self._state.chapter_attempts < 3:
            actions.append("classify_chapter")
        elif self._state.heading_score == 0.0 and self._state.heading_attempts < 3:
            actions.append("classify_heading")
        elif self._state.subheading_score < REWARD_WEIGHTS["subheading"] and self._state.subheading_attempts < 3:
            actions.append("classify_subheading")
        elif self._state.duty_score == 0.0 and self._state.duty_attempts < 3:
            actions.append("check_duty")
        elif self._state.sanctions_score == 0.0 and self._state.sanctions_attempts < 3:
            actions.append("check_sanctions")
        else:
            actions.append("submit")

        return actions
