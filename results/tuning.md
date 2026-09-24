| Model | params | precision@10 | recall@10 | ndcg@10 | coverage | seconds |
|---|---|---|---|---|---|---|
| ALS/confidence | `{}` | 0.1478 | 0.1879 | 0.2072 | 0.077 | 129.6 |
| ALS/confidence | `{"min_rating": 3.5}` | 0.1639 | 0.2029 | 0.2294 | 0.061 | 75.0 |
| ALS/confidence | `{"min_rating": 3.0, "graded": true}` | 0.1652 | 0.2036 | 0.2299 | 0.059 | 98.0 |
| ALS/confidence | `{"graded": true}` | 0.1575 | 0.1982 | 0.2201 | 0.069 | 134.3 |
| ALS/alpha | `{"min_rating": 3.0, "graded": true, "alpha": 3.0}` | 0.1614 | 0.1913 | 0.2230 | 0.043 | 112.1 |
| ALS/alpha | `{"min_rating": 3.0, "graded": true, "alpha": 10.0}` | 0.1652 | 0.2036 | 0.2299 | 0.059 | 115.1 |
| ALS/alpha | `{"min_rating": 3.0, "graded": true, "alpha": 30.0}` | 0.1502 | 0.1949 | 0.2126 | 0.085 | 122.9 |
| ALS/reg | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 0.1}` | 0.1652 | 0.2036 | 0.2299 | 0.059 | 121.9 |
| ALS/reg | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 5.0}` | 0.1656 | 0.2047 | 0.2311 | 0.059 | 121.6 |
| ALS/reg | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 50.0}` | 0.1655 | 0.2029 | 0.2300 | 0.056 | 111.2 |
| ALS/factors | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 5.0, "factors": 64}` | 0.1656 | 0.2047 | 0.2311 | 0.059 | 97.2 |
| ALS/factors | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 5.0, "factors": 128}` | 0.1524 | 0.1866 | 0.2116 | 0.070 | 135.8 |
| ALS/factors | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 5.0, "factors": 200}` | 0.1409 | 0.1707 | 0.1955 | 0.081 | 180.6 |
| ALS/cg_steps | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 5.0, "factors": 64, "cg_steps": 3}` | 0.1656 | 0.2047 | 0.2311 | 0.059 | 94.7 |
| ALS/cg_steps | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 5.0, "factors": 64, "cg_steps": 8}` | 0.1652 | 0.2033 | 0.2308 | 0.059 | 201.3 |
| ALS/pop_beta | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 5.0, "factors": 64, "cg_steps": 3, "pop_beta": 0.0}` | 0.1656 | 0.2047 | 0.2311 | 0.059 | 88.8 |
| ALS/pop_beta | `{"min_rating": 3.0, "graded": true, "alpha": 10.0, "reg": 5.0, "factors": 64, "cg_steps": 3, "pop_beta": 0.25}` | 0.1719 | 0.2101 | 0.2400 | 0.066 | 90.6 |
| EASE | `{"reg": 200.0}` | 0.1678 | 0.2095 | 0.2364 | 0.087 | 43.5 |
| EASE | `{"reg": 1000.0}` | 0.1718 | 0.2115 | 0.2404 | 0.075 | 43.5 |
| EASE | `{"reg": 4000.0}` | 0.1715 | 0.2107 | 0.2397 | 0.061 | 45.6 |
| EASE (>=3.5 stars) | `{"reg": 1000.0, "min_rating": 3.5}` | 0.1872 | 0.2238 | 0.2598 | 0.064 | 30.7 |
| BPR | `{}` | 0.1142 | 0.1365 | 0.1542 | 0.113 | 83.5 |
| BPR | `{"factors": 128, "epochs": 40}` | 0.1138 | 0.1368 | 0.1513 | 0.115 | 147.1 |
