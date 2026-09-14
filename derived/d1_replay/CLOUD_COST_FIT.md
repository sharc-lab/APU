# Cloud cost predictor (D-1c)

Generated: 2026-09-08T23:04:30.413905+00:00

- intercept: **0.010377**
- slope ($ / turn remaining): **0.071267**
- R²: **0.7132** (n=70 per-entry from_turn points)
- residual SE: **0.054412**
- tag: `ASSUMED(fit=20 entries cloud_multi_turn_report.json)`
- mean full-entry $: **0.270352**
- R0 @200 point: **$54.0704** [51.1228, 57.0180] (half-width $2.9476)

C-3 confirm: {"c3_fitted_exponent": 1.026974487407871, "c3_linear_r2_aggregated": 0.9798644526439462, "c3_verdict": "closer to linear in remaining cloud turns (fitted exponent 1.027); quadratic claim not supported", "linear_confirmed": true, "n_points_all_from_turn": 70, "n_points_full_entry": 20, "per_entry_all_from_turn_r2": 0.7132240894033286, "per_entry_full_entry_only_intercept": 0.045709673684210556, "per_entry_full_entry_only_r2": 0.4363912622899151, "per_entry_full_entry_only_slope": 0.06418357894736841}

