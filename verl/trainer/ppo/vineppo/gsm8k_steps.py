# SPDX-License-Identifier: MIT
# Copyright (c) 2024 McGill NLP
# Vendored from McGill-NLP/VinePPO@9f716e68bb934e71da86c226c1699c36b5553faf.
# See LICENSE.McGill for the original license.


def split_gsm8k(solution: str) -> list[int]:
    """
    Split the solution into reasoning steps.

    Args:
        solution: The solution text.

    Returns:
        A list of indices where each index corresponds to the start of a reasoning step.
        Example:
        >>> solution = '...'
        >>> indices = split_solution_into_reasoning_steps(solution)
        >>> steps = [solution[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]
    """

    delimiter = "\n"
    answer_prefix = "\n#### "

    if answer_prefix is None:
        sol_without_answer, answer = solution, None
    else:
        solution_parts = solution.split(answer_prefix)
        if len(solution_parts) < 2:
            sol_without_answer, answer = solution, None
        else:
            sol_without_answer, answer = solution_parts

    steps = sol_without_answer.split(delimiter)

    # Merge first empty steps to the first non-empty step
    first_non_empty_step_idx = None
    for i, step in enumerate(steps):
        if step.strip() != "":
            first_non_empty_step_idx = i
            break

    if first_non_empty_step_idx is not None and first_non_empty_step_idx > 0:
        new_first_step = delimiter.join(steps[: first_non_empty_step_idx + 1])

        steps = [new_first_step] + steps[first_non_empty_step_idx + 1 :]

    if answer is not None:
        # We want to merge the last step with the answer

        # Find last non-empty step index
        last_non_empty_step_idx = None
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].strip() != "":
                last_non_empty_step_idx = i
                break

        if last_non_empty_step_idx is None:
            # Then it means the entire solution is a single step
            last_non_empty_step_idx = 0

        new_last_step = delimiter.join(steps[last_non_empty_step_idx:])
        # Also merge the last step with the answer
        new_last_step = f"{new_last_step}{answer_prefix}{answer}"
        steps = steps[:last_non_empty_step_idx] + [new_last_step]

    reconstructed_solution = delimiter.join(steps)
    assert reconstructed_solution == solution, f"{reconstructed_solution} != {solution}"

    # Find the indices of the reasoning steps
    indices = [0]
    for i, step in enumerate(steps):
        if i == 0:
            indices.append(indices[-1] + len(step))
        else:
            indices.append(indices[-1] + len(step) + len(delimiter))

    assert indices[-1] == len(solution), f"{indices[-1]} != {len(solution)}"

    return indices
