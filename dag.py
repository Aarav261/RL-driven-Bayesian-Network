"""DAG mechanics for RLiG: adjacency matrix + legal edge edits.

This is step 1 of the build order and the foundation everything stands on.
Everything is a d x d numpy int matrix A where A[i, j] == 1 means edge i -> j.
"""

import numpy as np

ADD, DELETE, REVERSE = "add", "delete", "reverse"


def empty_dag(d):
    return np.zeros((d, d), dtype=int)


def is_acyclic(A):
    """Kahn's algorithm: peel off nodes with no incoming edges."""
    A = A.copy()
    indeg = A.sum(axis=0)
    frontier = [n for n in range(len(A)) if indeg[n] == 0]
    seen = 0
    while frontier:
        n = frontier.pop()
        seen += 1
        for j in np.where(A[n] == 1)[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                frontier.append(j)
    return seen == len(A)


def apply_action(A, action):
    """action = (op, i, j). Returns a NEW matrix; does not mutate A."""
    op, i, j = action
    B = A.copy()
    if op == ADD:
        B[i, j] = 1
    elif op == DELETE:
        B[i, j] = 0
    elif op == REVERSE:
        B[i, j] = 0
        B[j, i] = 1
    else:
        raise ValueError(f"unknown op {op}")
    return B


def all_actions(d):
    """Every candidate edit for a d-node graph (before legality filtering)."""
    acts = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            acts.append((ADD, i, j))
            acts.append((DELETE, i, j))
            acts.append((REVERSE, i, j))
    return acts


def is_legal(A, action, max_indegree):
    """Legal = well-formed edit that keeps a DAG under the in-degree cap.

    This is the mask that keeps the RL agent from wasting steps on penalties.
    """
    op, i, j = action
    if op == ADD:
        if A[i, j] == 1 or A[j, i] == 1:      # already an edge either way
            return False
        if A[:, j].sum() + 1 > max_indegree:  # too many parents for j
            return False
    elif op == DELETE:
        if A[i, j] == 0:
            return False
    elif op == REVERSE:
        if A[i, j] == 0:
            return False
        if A[:, i].sum() + 1 > max_indegree:  # i gains a parent
            return False
    return is_acyclic(apply_action(A, action))


def legal_action_mask(A, max_indegree):
    """Boolean list aligned with all_actions(d)."""
    return [is_legal(A, a, max_indegree) for a in all_actions(len(A))]


def _demo():
    # 3 nodes, in-degree cap 2.
    d, k = 3, 2
    A = empty_dag(d)
    assert is_acyclic(A)

    A = apply_action(A, (ADD, 0, 1))          # 0 -> 1
    A = apply_action(A, (ADD, 1, 2))          # 1 -> 2
    assert is_acyclic(A)
    assert A[0, 1] == 1 and A[1, 2] == 1

    # Adding 2 -> 0 would make a cycle 0->1->2->0.
    assert not is_legal(A, (ADD, 2, 0), k)
    # Adding the reverse 0 -> 2 is fine.
    assert is_legal(A, (ADD, 0, 2), k)
    # Re-adding an existing edge is illegal.
    assert not is_legal(A, (ADD, 0, 1), k)
    # Reversing 1 -> 2 to 2 -> 1 keeps acyclicity.
    assert is_legal(A, (REVERSE, 1, 2), k)

    # In-degree cap: give node 2 two parents, a third is blocked.
    A2 = apply_action(A, (ADD, 0, 2))         # now 0->2 and 1->2
    assert A2[:, 2].sum() == 2
    # No third node exists to parent 2 here, so check the cap logic directly:
    B = empty_dag(4)
    B = apply_action(B, (ADD, 0, 3))
    B = apply_action(B, (ADD, 1, 3))
    assert not is_legal(B, (ADD, 2, 3), max_indegree=2)
    assert is_legal(B, (ADD, 2, 3), max_indegree=3)

    # Mask length matches the action space.
    assert len(legal_action_mask(A, k)) == len(all_actions(d))
    print("dag.py self-check passed")


if __name__ == "__main__":
    _demo()
