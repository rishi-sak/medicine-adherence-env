def grade(task_name, state):

    total_meds = len(state.medicines)
    taken = sum(1 for m in state.medicines if m.taken)
    missed = state.missed_doses

    if task_name == "easy":
        # proportion of medicines taken
        return round(taken / total_meds, 2)

    elif task_name == "medium":
        # reward fewer missed doses + partial progress
        adherence_score = taken / total_meds
        penalty = missed * 0.2
        score = adherence_score - penalty
        return max(0.0, round(score, 2))

    elif task_name == "hard":
        # stricter scoring with higher penalty
        adherence_score = taken / total_meds
        penalty = missed * 0.3

        # bonus if all taken without miss
        if missed == 0 and taken == total_meds:
            return 1.0

        score = adherence_score - penalty
        return max(0.0, round(score, 2))