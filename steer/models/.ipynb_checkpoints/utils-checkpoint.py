"""
utils.py (steer/models/utils.py)
=================================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  add_vector_from_position: complete rewrite.

          Original behaviour:
            - from_pos=None was valid and applied vector to all positions
            - Handled 2D and 3D matrix shapes separately with explicit
              dtype casting per branch
            - Had a [DEBUG] print statement in the 2D branch
            - Raised ValueError when from_pos was None (contradicting
              the None=all-positions use case in AttnWrapper)

          Modified behaviour:
            - from_pos=None is now valid — defaults to all positions
              by using a ones mask
            - Unified dtype handling: cast everything to float() for the
              addition, then restore original dtype — works for all shapes
            - Removed the [DEBUG] print
            - from_pos=None no longer raises — instead applies to all
              positions (needed by AttnWrapper which calls this with
              from_position=None when no position offset is set)
            - Simplified: no explicit 2D/3D branching needed since
              broadcasting handles both cases

          This is the only change; get_a_b_probs and
          make_tensor_save_suffix are identical to original.
"""

import torch as t


def add_vector_from_position(matrix, vector, position_ids, from_pos=None):
    """Add `vector` to `matrix` at positions >= `from_pos`.

    Args:
        matrix:       Activation tensor [seq, hidden] or [batch, seq, hidden]
        vector:       Steering vector, broadcastable to matrix shape
        position_ids: Position id tensor matching matrix sequence dim, or None
        from_pos:     Minimum position to apply steering at, or None (= all)

    Returns:
        matrix + masked vector, same dtype as input matrix
    """
    orig_dtype = matrix.dtype

    if position_ids is None:
        # Apply to all positions
        mask = t.ones_like(matrix, dtype=orig_dtype)
    else:
        from_id = from_pos
        if from_id is None:
            from_id = position_ids.min().item() - 1  # effectively all positions

        mask = (position_ids >= from_id).unsqueeze(-1)  # [..., seq, 1]

    # Cast to float for numerically stable addition, restore original dtype
    matrix = matrix.float() + mask.float() * vector.float()
    return matrix.to(orig_dtype)


def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs  = t.softmax(last_token_logits, dim=-1)
    return last_token_probs[a_token_id].item(), last_token_probs[b_token_id].item()


def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'
