def predict_signal(row):
    # TODO: Use XGBoost or LSTM with real features
    import random
    return "BUY" if random.random() > 0.5 else "HOLD"