def auto_select_k(eigenvalues: list[float], threshold: float = 0.95) -> int:
    eigenvalues = sorted(eigenvalues, reverse=True)
    total = sum(eigenvalues)
    if total == 0:
        return len(eigenvalues)

    cumulative = 0
    for i, val in enumerate(eigenvalues):
        cumulative += val
        if cumulative / total >= threshold:
            return i + 1
    return len(eigenvalues)