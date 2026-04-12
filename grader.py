def grade(task_name, state):
    LO = 1e-6   # strictly above 0
    HI = 1 - 1e-6  # strictly below 1

    def clamp(x):
        return max(LO, min(HI, float(x)))

    total_meds = len(state.medicines)
    if total_meds == 0:
        return 0.5

    taken = sum(1 for m in state.medicines if m.taken)
    missed = state.missed_doses

    # Laplace smoothing — numerically safe, never hits 0 or 1
    adherence_score = (taken + 1) / (total_meds + 2)

    if task_name == "easy":
        score = adherence_score
    elif task_name == "medium":
        penalty = missed * 0.2
        score = adherence_score - penalty
    elif task_name == "hard":
        penalty = missed * 0.3
        score = adherence_score - penalty
    else:
        return 0.5

    return clamp(score)