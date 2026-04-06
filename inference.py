import os
from openai import OpenAI   
from env import MedicineEnv
from models import Action
from tasks import TASKS
from grader import grade

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy").lower()
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
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


def run_task(task_name):
    env = MedicineEnv(task=task_name)
    state = env.reset()

    log_start(task_name)

    rewards = []
    step_num = 0

    while True:
        step_num += 1

        # Simple deterministic agent
        action_obj = None
        for med in state.medicines:
            if not med.taken:
                action_obj = Action(
                    action_type="mark_taken",
                    medicine_name=med.name
                )
                break

        if action_obj is None:
            break

        result = env.step(action_obj)

        reward = result.reward
        done = result.done

        rewards.append(reward)

        log_step(step_num, action_obj.action_type, reward, done)

        state = result.observation

        if done:
            break

    #  Use grader
    score = grade(task_name, state)

    success = score > 0.5

    log_end(success, step_num, rewards)


def main():
    for task_name in TASKS.keys():
        run_task(task_name)


if __name__ == "__main__":
    main()