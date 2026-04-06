from models import Observation, Action, Medicine, StepResult


class MedicineEnv:

    def __init__(self, task: str = "easy"):
        self._state = None
        self.current_step = 0
        self.max_steps = 12   # ⬆️ increased steps
        self.task = task

    def reset(self) -> Observation:
        self.current_step = 0

        # ✅ Different difficulty levels
        if self.task == "easy":
            medicines = [
                Medicine(name="Insulin", time="08:00"),
                Medicine(name="Paracetamol", time="09:00"),
            ]

        elif self.task == "medium":
            medicines = [
                Medicine(name="Insulin", time="08:00"),
                Medicine(name="Paracetamol", time="10:00"),
                Medicine(name="Antibiotic", time="12:00"),
            ]

        else:  # hard
            medicines = [
                Medicine(name="Insulin", time="08:00"),
                Medicine(name="Paracetamol", time="10:00"),
                Medicine(name="Antibiotic", time="12:00"),
                Medicine(name="Vitamin", time="14:00"),
            ]

        self._state = Observation(
            current_time="08:00",
            medicines=medicines,
            missed_doses=0,
            patient_risk="high"
        )

        return self._state

    def state(self) -> Observation:
        return self._state

    def _advance_time(self):
        hour, minute = map(int, self._state.current_time.split(":"))
        hour += 1
        if hour >= 24:
            hour = 0
        self._state.current_time = f"{hour:02d}:{minute:02d}"

    def step(self, action: Action) -> StepResult:
        reward = 0.0
        done = False
        self.current_step += 1

        valid_actions = ["mark_taken", "send_reminder", "notify_caretaker", "reschedule"]

        if action.action_type not in valid_actions:
            return StepResult(
                observation=self._state,
                reward=-1.0,
                done=False,
                info={"error": "invalid_action"}
            )

        found = False

        for med in self._state.medicines:
            if med.name == action.medicine_name:
                found = True

                # ⛔ Taking medicine before time → penalty
                if action.action_type == "mark_taken":
                    if self._state.current_time < med.time:
                        reward -= 0.5  # early intake penalty
                    elif not med.taken:
                        med.taken = True
                        reward += 1.0
                    else:
                        reward -= 0.3

                elif action.action_type == "send_reminder":
                    if not med.taken:
                        reward += 0.2

                elif action.action_type == "notify_caretaker":
                    if not med.taken:
                        reward += 0.4

                elif action.action_type == "reschedule":
                    med.time = self._state.current_time
                    reward += 0.1

        if not found:
            reward -= 0.7

        # ⛔ Missed dose logic (stronger)
        for med in self._state.medicines:
            if not med.taken and med.time < self._state.current_time and not med.missed:
                med.missed = True
                self._state.missed_doses += 1

                reward -= 0.8  # stronger penalty

                if self._state.patient_risk == "high":
                    reward -= 0.7  # extra penalty

        # advance time
        self._advance_time()

        # done condition
        if all(m.taken for m in self._state.medicines):
            done = True

        if self.current_step >= self.max_steps:
            done = True
            reward -= 0.5

        return StepResult(
            observation=self._state,
            reward=round(reward, 2),
            done=done,
            info={
                "current_step": self.current_step,
                "missed_doses": self._state.missed_doses,
                "time": self._state.current_time
            }
        )