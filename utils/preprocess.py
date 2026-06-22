import numpy as np

def make_test_sequences(data, window_size, feature_cols):
    X = []

    for unit in sorted(data["unit_number"].unique()):
        unit_df = data[data["unit_number"] == unit].sort_values("Operation_cycle")
        features = unit_df[feature_cols].values

        if len(features) >= window_size:
            X.append(features[-window_size:])
        else:
            pad = np.zeros((window_size - len(features), len(feature_cols)))
            X.append(np.vstack([pad, features]))

    return np.array(X)