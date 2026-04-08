import json
import os
import random
import uuid
from openenv.core.env_server import Environment
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import CustomsAction, CustomsObservation, CustomsState

# ---------------------------------------------------------------------------
# Constants
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
    "task_easy":   {"max_steps": 10},
    "task_medium": {"max_steps": 20},
    "task_hard":   {"max_steps": 30},
}

TASK_BRIEF = (
    "Classify this shipment: determine the correct HS chapter, "
    "heading, and subheading, check the duty rate and sanctions "
    "status, then submit a final verdict of approve or hold."
)

SHIPMENTS = [
    # ── EASY (5) ── non-sanctioned, technically clear, learnable
    {
        "product_description": (
            "Acrylonitrile-butadiene-styrene (ABS) copolymer resin, injection-molding grade, "
            "melt flow index 22 g/10 min at 220 deg C/10kg, natural color, supplied in 25kg "
            "polyethylene-lined paper bags on standard pallets. "
            "Intended for automotive interior trim and consumer electronics enclosures."
        ),
        "country_of_origin": "Germany",
        "declared_value": 85000.0,
        "importer_name": "Polymat Engineering Pvt Ltd",
        "hs_chapter": "39",
        "hs_heading": "3903",
        "hs_subheading": "3903.30.00.00"
    },
    {
        "product_description": (
            "Men's knitted outer garments, long-sleeved, cut-and-sewn from 100% polyester fleece "
            "fabric weighing 280 g/m2, crew neck with ribbed cuffs and hem, quarter-zip front closure, "
            "containing 25 percent or more by weight of leather trim inserts, sizes M to 3XL, "
            "assorted colors, packed 6 pieces per polybag, 48 pieces per master carton."
        ),
        "country_of_origin": "Bangladesh",
        "declared_value": 14000.0,
        "importer_name": "Apex Garments Pvt Ltd",
        "hs_chapter": "61",
        "hs_heading": "6101",
        "hs_subheading": "6101.30.10.00"
    },
    {
        "product_description": (
            "Men's woven overcoat of cotton shell fabric, padded sleeveless jacket construction "
            "with quilted lining, detachable hood with drawcord, DWR-treated waterproof outer shell, "
            "sizes S to XXL, garment-washed and pre-shrunk. Style code JK-9211. FOB Osaka."
        ),
        "country_of_origin": "Japan",
        "declared_value": 22000.0,
        "importer_name": "Nippon Textile Imports",
        "hs_chapter": "62",
        "hs_heading": "6201",
        "hs_subheading": "6201.20.19.00"
    },
    {
        "product_description": (
            "Ferrosilicon alloy, silicon content 19.8% by weight, not more than 1 percent of carbon, "
            "standard lump size 10 to 100mm, total shipment 48 metric tons in 1-tonne big bags. "
            "Used as deoxidizer in electric arc furnace steelmaking. COA and mill certificate included."
        ),
        "country_of_origin": "Saudi Arabia",
        "declared_value": 48000.0,
        "importer_name": "Ferroalloys India Ltd",
        "hs_chapter": "72",
        "hs_heading": "7202",
        "hs_subheading": "7202.19.10.00"
    },
    {
        "product_description": (
            "Watertube steam boiler, natural circulation, rated steam output 68 tonnes per hour "
            "at 64 bar and 480 deg C, consisting of upper and lower drums, membrane wall panels, "
            "superheater, economizer and air preheater sections. "
            "Designed for combined heat and power plant service."
        ),
        "country_of_origin": "Taiwan",
        "declared_value": 310000.0,
        "importer_name": "Thermex Power Systems Pvt Ltd",
        "hs_chapter": "84",
        "hs_heading": "8402",
        "hs_subheading": "8402.11.00.00"
    },

    # ── MEDIUM (5) ── China/Vietnam/US, technically ambiguous
    {
        "product_description": (
            "Synchronous AC single-phase electric motor, rated output 0.25 kW at 1,450 rpm, "
            "50 Hz, 230V, class F insulation, IP44 enclosure, valued at USD 3.80 per unit. "
            "For use in household appliance drives. Quantity: 13,700 units. Model: SM-025-230-50."
        ),
        "country_of_origin": "China",
        "declared_value": 52000.0,
        "importer_name": "Dragon Motors India Pvt Ltd",
        "hs_chapter": "85",
        "hs_heading": "8501",
        "hs_subheading": "8501.10.20.00"
    },
    {
        "product_description": (
            "AC single-phase electric motor, 0.37 kW rated output at 2,850 rpm, 230V 50Hz, "
            "capacitor-start induction-run type, foot-mounted flange, class B insulation, IP54 rating, "
            "output exceeds 74.6 W but does not exceed 735 W. "
            "Application: water pump drives. Quantity: 800 units."
        ),
        "country_of_origin": "Vietnam",
        "declared_value": 38000.0,
        "importer_name": "Saigon Electric Components Ltd",
        "hs_chapter": "85",
        "hs_heading": "8501",
        "hs_subheading": "8501.20.40.00"
    },
    {
        "product_description": (
            "Three-phase AC induction motor, output 7.5 kW at 1,460 rpm, 400V delta/690V star, "
            "50Hz, IE3 efficiency class, frame size IEC 132M, foot and flange mounted, "
            "output exceeds 750 W but does not exceed 14.92 kW. CNAS test certificate enclosed. "
            "Application: industrial pump and fan drives. Quantity: 240 units."
        ),
        "country_of_origin": "China",
        "declared_value": 125000.0,
        "importer_name": "Sinotech Power Equipment India",
        "hs_chapter": "85",
        "hs_heading": "8501",
        "hs_subheading": "8501.32.20.00"
    },
    {
        "product_description": (
            "Super-heated water boiler, forced-circulation type, thermal output 14.8 MW, "
            "operating pressure 16 bar, maximum water temperature 130 deg C, natural gas fired burner, "
            "fully factory assembled and skid-mounted with controls, pumps and expansion vessel. "
            "For district heating system integration."
        ),
        "country_of_origin": "United States",
        "declared_value": 195000.0,
        "importer_name": "HeatTech Industrial India",
        "hs_chapter": "84",
        "hs_heading": "8402",
        "hs_subheading": "8402.20.00.00"
    },
    {
        "product_description": (
            "Step-index multimode optical fiber in continuous lengths, core diameter 62.5 micron, "
            "cladding diameter 125 micron, numerical aperture 0.275, attenuation 3.5 dB/km at 850nm, "
            "acrylate-coated, supplied on 4.4km wooden reels. "
            "For use in local area network backbone cabling."
        ),
        "country_of_origin": "Vietnam",
        "declared_value": 67000.0,
        "importer_name": "Photonix Components India Pvt Ltd",
        "hs_chapter": "90",
        "hs_heading": "9001",
        "hs_subheading": "9001.10.00"
    },

    # ── HARD (5) ── sanctioned countries, high dual-use risk
    {
        "product_description": (
            "Pressurized water nuclear reactor vessel assembly, VVER-1200 design, rated thermal output "
            "3,200 MW, forged low-alloy steel vessel body with stainless steel cladding, "
            "including upper and lower internals, control rod drive mechanism housings "
            "and in-core instrumentation thimbles. For Kudankulam Unit 5 nuclear power station."
        ),
        "country_of_origin": "Russia",
        "declared_value": 890000.0,
        "importer_name": "Atom Energy Equipment India",
        "hs_chapter": "84",
        "hs_heading": "8401",
        "hs_subheading": "8401.10.00.00"
    },
    {
        "product_description": (
            "Ferrosilicon manganese alloy, silicon content 17.4% and manganese content 65.2% by weight, "
            "lump size 20 to 80mm, total 120 metric tons in 1-tonne jumbo bags. "
            "Mill certificate and chemical analysis report included. "
            "Used in flat carbon steel production."
        ),
        "country_of_origin": "Iran",
        "declared_value": 55000.0,
        "importer_name": "Tehran Alloys India Imports",
        "hs_chapter": "72",
        "hs_heading": "7202",
        "hs_subheading": "7202.30.00.00"
    },
    {
        "product_description": (
            "Surface condenser for steam turbine exhaust, shell-and-tube type, "
            "cooling surface area 4,200 m2, titanium Grade 2 tubes, carbon steel shell, "
            "design pressure 0.15 bar absolute, condensate extraction pump and air ejector system included. "
            "For 300MW thermal power unit."
        ),
        "country_of_origin": "North Korea",
        "declared_value": 420000.0,
        "importer_name": "Pyongyang Industrial Imports",
        "hs_chapter": "84",
        "hs_heading": "8404",
        "hs_subheading": "8404.20.00.00"
    },
    {
        "product_description": (
            "Styrene-acrylonitrile (SAN) copolymer resin in pellet form, acrylonitrile content "
            "24.5% by weight, transparency grade, melt flow rate 8 g/10 min, "
            "Vicat softening point 103 deg C, packed in 25kg moisture-proof bags. "
            "For optical quality injection-molded parts and display lenses."
        ),
        "country_of_origin": "China",
        "declared_value": 78000.0,
        "importer_name": "ChemTech Polymers India",
        "hs_chapter": "39",
        "hs_heading": "3903",
        "hs_subheading": "3903.20.00.00"
    },
    {
        "product_description": (
            "Electromagnetic isotope separation apparatus, calutron-type ion source assembly, "
            "including vacuum chamber, magnet yoke, ion beam collector arrays "
            "and high-voltage power supply regulation units. "
            "Rated separation capacity 0.4 kg/day of stable isotope product, "
            "with associated control instrumentation and vacuum pumping systems."
        ),
        "country_of_origin": "Syria",
        "declared_value": 640000.0,
        "importer_name": "Damascus Tech Industries India",
        "hs_chapter": "84",
        "hs_heading": "8401",
        "hs_subheading": "8401.20.00.00"
    },
]



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    hts_path = os.path.join(data_dir, "hts_data.json")
    with open(hts_path, "r", encoding="utf-8") as f:
        hts_raw = json.load(f)

    # Build lookup: subheading -> {description, general (duty rate)}
    hts_lookup = {}
    for entry in hts_raw:
        key = entry["htsno"].strip()
        hts_lookup[key] = {
            "description": entry["description"],
            "duty_rate": entry["general"]
        }

    ofac_path = os.path.join(data_dir, "ofac_sdn.csv")
    sanctioned = set()
    with open(ofac_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:  # skip header
        country = line.strip().strip('"')
        if country:
            sanctioned.add(country.upper())

    return hts_lookup, sanctioned


HTS_LOOKUP, SANCTIONED_COUNTRIES = load_data()

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

# Countries definitively sanctioned by US OFAC.
# Our OFAC CSV uses program codes (DPRK, IRAN, RUSSIA) not common names,
# so we maintain this authoritative set for reliable lookup.
HARDCODED_SANCTIONED = {
    "NORTH KOREA", "DPRK", "IRAN", "RUSSIA", "SYRIA",
    "BELARUS", "CUBA", "VENEZUELA", "MYANMAR", "SUDAN",
    "ZIMBABWE", "LIBYA", "SOMALIA", "IRAQ", "LEBANON",
}


def is_sanctioned(country: str) -> bool:
    country_upper = country.upper().strip()

    # 1. Hardcoded set — most reliable, handles DPRK/North Korea alias
    if country_upper in HARDCODED_SANCTIONED:
        return True

    # 2. Direct lookup in OFAC CSV data
    if country_upper in SANCTIONED_COUNTRIES:
        return True

    # 3. Token match — catches "DPRK2", "CAATSA - RUSSIA", etc.
    for entry in SANCTIONED_COUNTRIES:
        for token in HARDCODED_SANCTIONED:
            if token in entry and country_upper in token:
                return True

    return False



# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomsEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = CustomsState()
        self._rng = random.Random()
        self._prev_score = 0.0
        self._current_shipment = {}

    def reset(self, seed=None, episode_id=None, task_id="task_easy", **kwargs) -> CustomsObservation:

        seed = seed if seed is not None else random.randint(0, 99999)
        self._rng = random.Random(seed)
        self._prev_score = 0.0

        # Pick shipment based on task_id
        if task_id == "task_easy":
            candidates = [s for s in SHIPMENTS if not is_sanctioned(s["country_of_origin"])]
        elif task_id == "task_medium":
            candidates = SHIPMENTS
        else:
            candidates = [s for s in SHIPMENTS if is_sanctioned(s["country_of_origin"])]
            if not candidates:
                candidates = SHIPMENTS

        shipment = self._rng.choice(candidates)
        self._current_shipment = shipment

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
            seed=seed,
            task_id=task_id,
            max_steps=max_steps,
        )

        shipment_text = (
            f"Product: {shipment['product_description']}\n"
            f"Country of Origin: {shipment['country_of_origin']}\n"
            f"Declared Value: USD {shipment['declared_value']:,.2f}\n"
            f"Importer: {shipment['importer_name']}"
        )

        return CustomsObservation(
            done=False,
            reward=None,
            shipment_description=shipment_text,
            feedback="New shipment received. Begin classification with classify_chapter.",
            available_actions=["classify_chapter"],
            task_brief=TASK_BRIEF,
            step_budget_remaining=max_steps,
            current_score=0.0,
        )

    def step(self, action: CustomsAction, timeout_s=None, **kwargs) -> CustomsObservation:  # type: ignore[override]

        self._state.step_count += 1
        self._state.current_step += 1
        steps_used = self._state.current_step
        max_steps = self._state.max_steps
        steps_remaining = max(0, max_steps - steps_used)

        action_type = action.action_type.strip().lower()
        value = action.value.strip()

        feedback = ""
        reward = 0.0

        if action_type == "classify_chapter":
            if self._state.chapter_score == 0.0:
                if value == self._state.correct_chapter:
                    self._state.chapter_score = REWARD_WEIGHTS["chapter"]
                    feedback = f"Correct. Chapter {value} confirmed. Proceed to classify_heading."
                else:
                    feedback = f"Incorrect chapter. Review the product category. Expected chapter: {self._state.correct_chapter[:1]}X"
            else:
                feedback = "Chapter already classified. Proceed to classify_heading."
            next_actions = ["classify_heading"]

        elif action_type == "classify_heading":
            if self._state.heading_score == 0.0:
                if value == self._state.correct_heading:
                    self._state.heading_score = REWARD_WEIGHTS["heading"]
                    feedback = f"Correct. Heading {value} confirmed. Proceed to classify_subheading."
                else:
                    feedback = f"Incorrect heading. Review the product type within chapter {self._state.correct_chapter}."
            else:
                feedback = "Heading already classified. Proceed to classify_subheading."
            next_actions = ["classify_subheading"]

        elif action_type == "classify_subheading":
            if self._state.subheading_score == 0.0:
                agent_sub = value.replace(" ", "").replace("-", "")
                correct_sub = self._state.correct_subheading.replace(" ", "").replace("-", "")
                if agent_sub == correct_sub or value == self._state.correct_subheading:
                    self._state.subheading_score = REWARD_WEIGHTS["subheading"]
                    feedback = f"Correct. Subheading {value} confirmed. Proceed to check_duty."
                else:
                    feedback = "Incorrect subheading. Narrow further within the heading."
            else:
                feedback = "Subheading already classified. Proceed to check_duty."
            next_actions = ["check_duty"]

        elif action_type == "check_duty":
            if self._state.duty_score == 0.0:
                agent_duty = value.lower().strip().replace(" ", "").replace("%", "")
                correct_duty = self._state.correct_duty_rate.lower().strip().replace(" ", "").replace("%", "")
                if agent_duty == correct_duty or agent_duty in correct_duty or correct_duty in agent_duty:
                    self._state.duty_score = REWARD_WEIGHTS["duty"]
                    feedback = f"Correct. Duty rate {self._state.correct_duty_rate} confirmed. Proceed to check_sanctions."
                else:
                    feedback = "Incorrect duty rate. Check the general rate column for this subheading."
            else:
                feedback = "Duty already checked. Proceed to check_sanctions."
            next_actions = ["check_sanctions"]

        elif action_type == "check_sanctions":
            if self._state.sanctions_score == 0.0:
                agent_val = value.lower().strip()
                if self._state.sanctions_hit:
                    if agent_val in ["flagged", "sanctioned", "yes", "true", "hold"]:
                        self._state.sanctions_score = REWARD_WEIGHTS["sanctions"]
                        feedback = "Correct. Country is on the OFAC sanctions list. Proceed to submit."
                    else:
                        feedback = "Incorrect. This country appears on the OFAC sanctions list."
                else:
                    if agent_val in ["clear", "clean", "no", "false", "approve", "not sanctioned"]:
                        self._state.sanctions_score = REWARD_WEIGHTS["sanctions"]
                        feedback = "Correct. Country is clear of sanctions. Proceed to submit."
                    else:
                        feedback = "Incorrect. This country is not on the OFAC sanctions list."
            else:
                feedback = "Sanctions already checked. Proceed to submit."
            next_actions = ["submit"]

        elif action_type == "submit":
            if self._state.verdict_score == 0.0:
                agent_verdict = value.lower().strip()
                if agent_verdict == self._state.correct_verdict:
                    self._state.verdict_score = REWARD_WEIGHTS["verdict"]
                    feedback = f"Correct verdict: {self._state.correct_verdict}. Classification complete."
                else:
                    feedback = "Incorrect verdict. Review your sanctions check."
            else:
                feedback = "Already submitted."
            next_actions = []

        else:
            feedback = f"Unknown action_type: {action_type}. Use one of the available_actions."
            next_actions = ["classify_chapter"]

        # Compute total score and step reward (delta)
        total_score = (
            self._state.chapter_score +
            self._state.heading_score +
            self._state.subheading_score +
            self._state.duty_score +
            self._state.sanctions_score +
            self._state.verdict_score
        )
        reward = round(total_score - self._prev_score, 4)
        self._prev_score = total_score

        # Episode ends if submit was called or budget exhausted
        done = (action_type == "submit") or (steps_remaining <= 0)

        shipment_text = (
            f"Product: {self._state.product_description}\n"
            f"Country of Origin: {self._state.country_of_origin}\n"
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
        )

    @property
    def state(self) -> CustomsState:
        return self._state
