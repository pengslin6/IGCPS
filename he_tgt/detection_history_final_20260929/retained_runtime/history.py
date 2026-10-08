from collections import deque
import numpy as np
LOOKBACK_NS=30_000_000_000
HISTORY=8

def make_history(values, times, streams, raw_indices, query_indices):
    """All inputs here are allowed context rows; query labels are absent."""
    assert np.all(np.diff(times) >= 0)
    lookup = {int(raw): i for i, raw in enumerate(query_indices)}
    n, width = len(query_indices), values.shape[1]
    current = np.zeros((n, width), dtype=np.float32)
    temporal = np.zeros((n, width*4+3), dtype=np.float32)
    history_ids = np.full((n, HISTORY), -1, dtype=np.int64)
    history_ages = np.full((n, HISTORY), -1., dtype=np.float64)
    found = np.zeros(n, dtype=bool)
    past = {}
    boundaries = np.r_[0, np.flatnonzero(np.diff(times)) + 1, len(times)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        now = int(times[lo])
        # Same-time observations cannot enter each other's histories.
        for row in range(lo, hi):
            if int(raw_indices[row]) not in lookup:
                continue
            target = lookup[int(raw_indices[row])]
            found[target] = True
            history = past.setdefault(int(streams[row]), deque(maxlen=HISTORY))
            while history and now-int(times[history[0]]) > LOOKBACK_NS:
                history.popleft()
            current[target] = values[row]
            temporal[target, :width] = values[row]
            if history:
                ids = np.array(history, dtype=np.int64)
                assert (times[ids] < now).all()
                age = (now-times[ids])/1e9
                temporal[target, width:2*width] = values[row]-values[ids[-1]]
                temporal[target, 2*width:3*width] = values[row]-values[ids].mean(0)
                temporal[target, 3*width:4*width] = values[ids].std(0)
                temporal[target, -3:] = [len(ids)/HISTORY, np.log1p(age[-1]), np.log1p(age[0])]
                history_ids[target, :len(ids)] = raw_indices[ids]
                history_ages[target, :len(ids)] = age
        for row in range(lo, hi):
            past.setdefault(int(streams[row]), deque(maxlen=HISTORY)).append(row)
    assert found.all()
    return {"static": current, "history": temporal, "history_indices": history_ids, "history_ages": history_ages,
        "raw_indices": np.asarray(query_indices, dtype=np.int64)}


