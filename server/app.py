from fastapi import FastAPI
import uvicorn

from env import MedicineEnv
from models import Action

app = FastAPI()
env = MedicineEnv(task="easy")  #  default task


@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "observation": state.model_dump(),
        "done": False
    }


@app.post("/step")
def step(action: Action):
    try:
        result = env.step(action)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/state")
def state():
    if env._state is None:
        return {"error": "Environment not initialized. Call /reset first"}
    return env.state().model_dump()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()