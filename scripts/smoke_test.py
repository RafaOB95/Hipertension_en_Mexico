# scripts/smoke_test.py
import pandas as pd
import numpy as np
from rafads.data import prepare_xy, train_val_test_split
from rafads.model import make_rf_pipeline, fit_model, predict_proba, predict_label
from rafads.evaluation import compute_metrics

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "edad": rng.integers(18, 85, size=n),
    "bmi": rng.normal(28, 5, size=n),
    "fumador": rng.choice(["si", "no"], size=n, p=[0.3, 0.7]),
    "sal": rng.choice(["alta", "media", "baja"], size=n, p=[0.2, 0.5, 0.3]),
})
logit = -8 + 0.06*df["edad"] + 0.12*df["bmi"] + (df["fumador"]=="si").astype(int)*0.8
p = 1 / (1 + np.exp(-logit))
df["hipertension"] = (rng.random(n) < p).astype(int)

X, y, pre, _, _ = prepare_xy(df, target="hipertension")
X_tr, X_te, y_tr, y_te = train_val_test_split(X, y)

pipe = make_rf_pipeline(pre, n_estimators=300, class_weight="balanced_subsample", random_state=42)
pipe = fit_model(pipe, X_tr, y_tr)

y_proba = predict_proba(pipe, X_te)
y_pred  = predict_label(pipe, X_te)
metrics = compute_metrics(y_te, y_proba, y_pred)

print("OK: pipeline entrenado.")
print("Métricas:", {k: metrics[k] for k in ("auc","accuracy","precision","recall","f1")})
