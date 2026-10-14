"""Use the same math-verification rule as the critic data collection."""
from examples.delta_critic.sampling_io import score_text


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    score = score_text(solution_str, str(ground_truth), {'method': 'math_verify', 'fallback_on_import_error': False})
    return {'score': score, 'acc': score}
