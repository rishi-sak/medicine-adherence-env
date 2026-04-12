def grade(task_name, state):
    EPS = 1e-3  # slightly larger epsilon to avoid precision issues

    total_meds = len(state.medicines)
    if total_meds == 0:
        return 0.5  # safe fallback

    taken = sum(1 for m in state.medicines if m.taken)
    missed = state.missed_doses

    # Safe adherence score (never exactly 0 or 1)
    adherence_score = (taken + EPS) / (total_meds + 2 * EPS)

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

    # Final strict clamp (NO rounding)
    if score <= 0:
        return EPS
    if score >= 1:
        return 1 - EPS

    return score