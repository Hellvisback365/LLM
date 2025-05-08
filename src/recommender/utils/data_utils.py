import numpy as np

def convert_numpy_types_for_json(obj):
    """Converte tipi numpy in tipi Python nativi per serializzazione JSON."""
    if isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types_for_json(i) for i in obj]
    return obj 