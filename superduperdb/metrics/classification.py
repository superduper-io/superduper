def compute_classification_metrics(validation_data, model, training_keys, metrics):
    X, y = training_keys
    out = {}
    predictions = model.predict([r[X] for r in validation_data])
    targets = [r[y] for r in validation_data]
    for m in metrics:
        out[m.identifier] = sum(
            [m(pred, target) for pred, target in zip(predictions, targets)]
        ) / len(validation_data)
    return out
