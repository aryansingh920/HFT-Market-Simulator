def adjust_transition_probabilities(config, steps_per_day):
    adjusted_regimes = []
    original_transitions = config['original_transitions']

    for regime_name in original_transitions:
        transitions = original_transitions[regime_name]
        adjusted_transitions = {}

        for target, prob_per_day in transitions.items():
            # Convert daily probability to per-step probability
            prob_per_step = prob_per_day / steps_per_day
            adjusted_transitions[target] = round(prob_per_step, 6)

        # Ensure staying probability is calculated correctly
        total_transition = sum(adjusted_transitions.values())
        adjusted_transitions[regime_name] = max(0, 1 - total_transition)

        # Get original regime parameters
        original_regime = next(
            r for r in config['regimes'] if r['name'] == regime_name)
        adjusted_regimes.append({
            'name': regime_name,
            'drift': original_regime['drift'],
            'vol_scale': original_regime['vol_scale'],
            'transitions': adjusted_transitions
        })

    config['regimes'] = adjusted_regimes
