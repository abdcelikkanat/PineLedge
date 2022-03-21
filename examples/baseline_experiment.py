exp = Experiments(seed=seed)
exp.set_events(events=train_events)

F = exp.get_freq_map()
simon_auc = roc_auc_score(y_true=train_labels, y_score=[F[sample[0], sample[1]] for sample in train_samples])
print(f"Simmon train auc: {simon_auc}")
simon_auc = roc_auc_score(y_true=test_labels, y_score=[F[sample[0], sample[1]] for sample in test_samples])
print(f"Simmon test auc: {simon_auc}")