---
title: Medicine Adherence Environment
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

#  Medicine Adherence Environment (OpenEnv)

##  Overview

This project implements a **real-world reinforcement learning environment** that simulates **patient medicine adherence behavior**.

The environment models how an AI agent can:

- Remind patients to take medicines

- Track missed doses

- Handle real-world timing constraints

- Optimize adherence for better health outcomes

---

##  Motivation

Medication non-adherence is a major real-world problem, especially among:

- Elderly patients

- Chronic illness patients

- High-risk individuals

This environment allows AI agents to learn:

- When to act (timing-sensitive decisions)

- How to balance reminders vs actions

- How to minimize missed doses

 It closely mimics real-world healthcare assistant systems.

---

##  Environment Design

The environment follows the standard OpenEnv API:

- `reset()` → Initialize environment  

- `step(action)` → Apply action and get result  

- `state()` → Get current state

---

##  Observation Space

```json

{

  "current_time": "string",

  "medicines": [

    {

      "name": "string",

      "time": "string",

      "taken": "boolean",

      "missed": "boolean"

    }

  ],

  "missed_doses": "integer",

  "patient_risk": "low | medium | high"

} 


 Action Space
---------------

{\
  "action_type": "mark_taken | send_reminder | notify_caretaker | reschedule",\
  "medicine_name": "string"\
}

### Actions:

-   **mark_taken** → Mark medicine as consumed
-   **send_reminder** → Remind patient
-   **notify_caretaker** → Escalate to caretaker
-   **reschedule** → Adjust medicine timing

 Tasks & Difficulty Levels
----------------------------

###  Easy

-   Few medicines
-   Simple timing
-   Minimal penalties

 Goal: Take all medicines on time

* * * * *

###  Medium

-   More medicines
-   Delayed schedules
-   Moderate penalties

 Goal: Handle missed doses efficiently

* * * * *

###  Hard

-   Multiple medicines
-   Strict timing constraints
-   Strong penalties for mistakes

 Goal: Optimize adherence and minimize missed doses

* * * * *

 Reward Design
----------------

-    Correct action → Positive reward
-    Early intake → Penalty
-    Missed dose → Strong penalty
-    High-risk patient → Extra penalty

 Encourages **timing-aware decision making**

* * * * *

 Setup Instructions
---------------------

### 1\. Clone repository

git clone <your-repo-url>\
cd medicine_env

* * * * *

### 2\. Install dependencies

pip install -r requirements.txt

* * * * *

### 3\. Run environment locally

py inference.py

* * * * *

### 4\. Run API server

python server/app.py

Then open:

http://localhost:7860/docs

* * * * *

 Docker Usage
---------------

docker build -t medicine-env .\
docker run -p 7860:7860 medicine-env

* * * * *

 Baseline Performance
-----------------------

| Task  | Steps | Rewards                                     |
| ----  | ---   | ----------------------------------------    |
| Easy  | 2     | 1.00, 1.00                                  |
| Medium| 5     | 1.00, -0.50, 1.00, -0.50, 1.00              |
| Hard  | 7     | 1.00, -0.50, 1.00, -0.50, 1.00, -0.50, 1.00 |

 Shows increasing difficulty and decision complexity

* * * * *

 Key Features
---------------

-    Time-aware environment
-    Real-world healthcare simulation
-    Multi-task difficulty levels
-    Reward shaping for learning
-    OpenEnv compatible
-    API + Docker ready

* * * * *

 Conclusion
-------------

This environment provides a **realistic, scalable, and structured RL problem** where agents must learn:

-   When to act
-   What action to take
-   How to minimize risk

 Making it ideal for training **intelligent healthcare assistants**

* * * * *

 Author
------------

Rishi, Tanishq, Akash