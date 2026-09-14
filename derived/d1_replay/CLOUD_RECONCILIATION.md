# Cloud cost exact-lookup test (D-1c; not H-1 reported number)

Generated: 2026-09-08T23:04:30.412904+00:00

sum over API calls of cloud_usd(prompt_tokens, completion_tokens); each call resent full context; tokens from cloud report, not local trace

Sealed 20-entry token lookup; H-1 reports the fit predictor, not this.

Predicted total **$5.407044** vs measured **$5.407044** (sealed spend $5.407044). Max |error| **$1.11022e-16**.

| id | predicted $ | measured $ | error $ | n_calls | prompt tok | completion tok |
|---|---:|---:|---:|---:|---:|---:|
| multi_turn_base_0 | 0.528648 | 0.528648 | 1.11e-16 | 16 | 166296 | 1984 |
| multi_turn_base_1 | 0.201192 | 0.201192 | 0 | 11 | 61849 | 1043 |
| multi_turn_base_2 | 0.407508 | 0.407508 | -1.11e-16 | 15 | 128736 | 1420 |
| multi_turn_base_3 | 0.108381 | 0.108381 | 1.39e-17 | 6 | 32272 | 771 |
| multi_turn_base_4 | 0.264501 | 0.264501 | -5.55e-17 | 8 | 82487 | 1136 |
| multi_turn_base_5 | 0.384318 | 0.384318 | 0 | 12 | 121686 | 1284 |
| multi_turn_base_6 | 0.288651 | 0.288651 | -5.55e-17 | 15 | 90592 | 1125 |
| multi_turn_base_7 | 0.219858 | 0.219858 | 2.78e-17 | 7 | 70206 | 616 |
| multi_turn_base_8 | 0.314790 | 0.314790 | 0 | 9 | 96935 | 1599 |
| multi_turn_base_9 | 0.250404 | 0.250404 | 0 | 13 | 75438 | 1606 |
| multi_turn_base_10 | 0.346032 | 0.346032 | 0 | 18 | 107054 | 1658 |
| multi_turn_base_11 | 0.160929 | 0.160929 | 0 | 5 | 50233 | 682 |
| multi_turn_base_12 | 0.157467 | 0.157467 | 5.55e-17 | 9 | 50034 | 491 |
| multi_turn_base_13 | 0.223779 | 0.223779 | 0 | 7 | 70353 | 848 |
| multi_turn_base_14 | 0.273201 | 0.273201 | 5.55e-17 | 10 | 84707 | 1272 |
| multi_turn_base_15 | 0.314784 | 0.314784 | -5.55e-17 | 10 | 99128 | 1160 |
| multi_turn_base_16 | 0.200511 | 0.200511 | 0 | 11 | 62122 | 943 |
| multi_turn_base_17 | 0.287127 | 0.287127 | 0 | 11 | 90884 | 965 |
| multi_turn_base_18 | 0.273666 | 0.273666 | 0 | 8 | 82687 | 1707 |
| multi_turn_base_19 | 0.201297 | 0.201297 | 2.78e-17 | 8 | 63459 | 728 |

