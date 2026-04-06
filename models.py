from pydantic import BaseModel
from typing import List, Optional, Dict, Literal


class Medicine(BaseModel):
    name: str
    time: str
    taken: bool = False
    missed: bool = False


class Observation(BaseModel):
    current_time: str
    medicines: List[Medicine]
    missed_doses: int = 0
    patient_risk: Literal["low", "medium", "high"]


class Action(BaseModel):
    action_type: Literal[
        "send_reminder",
        "mark_taken",
        "reschedule",
        "notify_caretaker"
    ]
    medicine_name: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Optional[Dict] = None