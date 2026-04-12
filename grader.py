def grade(task_name, state):
    EPS = 1e-6  # small value to keep score inside (0,1)

    total_meds = len(state.medicines)
    if total_meds == 0:
        return 0.5  # safe fallback (inside range)

    taken = sum(1 for m in state.medicines if m.taken)
    missed = state.missed_doses

    def clamp(score):
        # Handle invalid values
        if score is None:
            return 0.5

        # First clamp raw value
        if score <= 0:
            score = EPS
        elif score >= 1:
            score = 1 - EPS

        # Then round
        score = round(score, 4)

        # Final safety clamp after rounding
        if score <= 0:
            return EPS
        if score >= 1:
            return 1 - EPS

        return score

    if task_name == "easy":
        score = taken / total_meds
        return clamp(score)

    elif task_name == "medium":
        adherence_score = taken / total_meds
        penalty = missed * 0.2
        score = adherence_score - penalty
        return clamp(score)

    elif task_name == "hard":
        adherence_score = taken / total_meds
        penalty = missed * 0.3

        score = adherence_score - penalty

        # even perfect case must not be 1.0
        if missed == 0 and taken == total_meds:
            score = 1 - EPS

        return clamp(score)