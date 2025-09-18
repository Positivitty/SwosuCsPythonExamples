from typing import List, Sequence, Union

Number = Union[int, float]

def matmul(A: Sequence[Sequence[Number]], B: Sequence[Sequence[Number]]) -> List[List[Number]]:
    # Basic shape checks (ragged rows not allowed)
    if not A or not B:
        raise ValueError("A and B must be non-empty.")
    m, kA = len(A), len(A[0])
    kB, n = len(B), len(B[0])
    if any(len(row) != kA for row in A):
        raise ValueError("All rows of A must have the same length.")
    if any(len(row) != n for row in B):
        raise ValueError("All rows of B must have the same length.")
    if kA != kB:
        raise ValueError(f"Incompatible shapes: A is {m}×{kA}, B is {kB}×{n} (inner dims must match).")

    # C = A * B  (m×n)
    C = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        # Small cache-friendly optimization: accumulate by shared dim t
        for t in range(kA):
            a_it = A[i][t]
            if a_it == 0:
                continue
            Bt = B[t]
            for j in range(n):
                C[i][j] += a_it * Bt[j]
    return C

# Example
A = [[1, 2, 3],
     [4, 5, 6]]            # 2×3
B = [[7, 8],
     [9, 10],
     [11, 12]]            # 3×2
print(matmul(A, B))       # → [[58, 64], [139, 154]]
