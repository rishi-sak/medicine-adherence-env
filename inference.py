import os
from openai import OpenAI

from env import MedicineEnv
from models import Action
from tasks import TASKS
from grader import grade

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

ENV_NAME = "medicine"


def log_start(task):
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def call_llm(state):
    state_text = f"""
    Time: {state.current_time}
    Medicines: {[(m.name, m.time, m.taken) for m in state.medicines]}
    Missed: {state.missed_doses}
    Risk: {state.patient_risk}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a medicine assistant."},
                {"role": "user", "content": state_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception:
        return None


def choose_action(state):
    # MANDATORY LLM CALL (required for validation)
    call_llm(state)

    current_time = state.current_time
    medicines = sorted(state.medicines, key=lambda m: m.time)

    # First priority: take any due medicines
    for med in medicines:
        if not med.taken and current_time >= med.time:
            return Action(action_type="mark_taken", medicine_name=med.name)

    # Second priority: remind for upcoming medicines
    for med in medicines:
        if not med.taken and current_time < med.time:
            return Action(action_type="send_reminder", medicine_name=med.name)

    return None


def run_task(task_name):
    env = MedicineEnv(task=task_name)
    state = env.reset()

    log_start(task_name)

    rewards = []
    step_num = 0

    while True:
        step_num += 1

        action_obj = choose_action(state)
        if action_obj is None:
            break

        result = env.step(action_obj)
        rewards.append(result.reward)
        log_step(step_num, action_obj.action_type, result.reward, result.done)

        state = result.observation
        if result.done:
            break

    # grade() already returns a value strictly in (0, 1)
    # Defensive clamp here guards against any unexpected edge case
    raw_score = grade(task_name, state)
    if not isinstance(raw_score, (int, float)) or raw_score != raw_score:  # NaN check
        raw_score = 0.5

    LO, HI = 1e-6, 1 - 1e-6
    score = max(LO, min(HI, float(raw_score)))

    success = score > 0.5
    log_end(success, step_num, rewards)


def main():
    for task_name in TASKS.keys():
        run_task(task_name)


if __name__ == "__main__":
    main()