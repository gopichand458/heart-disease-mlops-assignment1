# Video script (target 6-8 minutes)

**How to record:** On your Mac press Cmd+Shift+5 -> "Record Selected Portion" (or the whole screen) -> Options -> choose
the microphone. Record the browser with the VM tab. Practise once; do it in 2-3 segments if easier (QuickTime can join
clips) and keep the terminal font large (View > zoom). Have everything pre-run, so the video shows *executing* only
the quick steps and *showing* the long ones (e.g. training results already printed).

| Time | Show | Say (in your own words) |
|---|---|---|
| 0:00-0:30 | Architecture diagram (`docs/architecture.png`) | Problem, dataset, the pipeline end to end |
| 0:30-1:15 | VS Code/terminal: repo tree (`tree -L 2`) | Where each assignment task lives |
| 1:15-2:15 | EDA figures / notebook | Class balance, missing values (`ca`, `thal`), strongest correlations, one insight |
| 2:15-3:30 | `python -m src.train` output + MLflow UI (runs, compare, plots) | Pipeline design, 3 models, CV vs hold-out, why the winner was selected, what is logged |
| 3:30-4:15 | `pytest`, `flake8`; GitHub Actions green run + artifacts; the red failure run | Test coverage, jobs, pipeline fails on errors |
| 4:15-5:15 | `make docker-build`, `docker run`, `smoke_test.sh`, Swagger `/docs` | Container, /predict contract, validation (422), JSON logs |
| 5:15-6:30 | `minikube kubectl -- get pods,svc`, tunnel/URL, curl through the service | Deployment manifest, 2 replicas, probes, LoadBalancer |
| 6:30-7:30 | Run `generate_traffic.py`, Prometheus target, Grafana dashboard moving | Metrics exposed, what each panel means, how you'd alert |
| 7:30-8:00 | Repo / report | Summary, limitations (303 rows, no drift monitoring yet), future work |

Tips: say *why* for each decision (that is the originality marks), mention one thing that went wrong and how you fixed it,
and keep the Grafana tab ready with traffic running before you start that segment. Export as `.mp4`.
