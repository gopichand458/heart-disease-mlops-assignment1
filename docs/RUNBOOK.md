# Runbook - executing the assignment on the Linux lab VM

Your VM (checked): Python 3.12, Docker 28, Minikube 1.35, sudo available, 4 CPU / 15 GB RAM.
**No `kubectl`/`kind`** - the scripts use `minikube kubectl --`.

**Rules of the VM:** work only under `~/workspace` (everything else is wiped on stop), keep the
lab browser tab open, and don't sit idle >15 min. **GitHub is your backup: push often.**

Take screenshots on your **Mac** (Cmd+Shift+4) of the VM browser tab, and save them with the exact
file names in section 8.

---
## 0. On your Mac (5 min): create the repo and push
```bash
cd "<unzipped folder>/heart-disease-mlops-pipeline"
git init -b main && git add . && git commit -m "Initial MLOps pipeline scaffold"
# create an EMPTY repo named heart-disease-mlops-pipeline on github.com first, then:
git remote add origin https://github.com/<you>/heart-disease-mlops-pipeline.git
git push -u origin main
```
Pushing triggers the first CI run (it downloads the data itself). If the repo is private, make it
public or add the evaluator as a collaborator.

## 1. In the VM: clone and set up (10 min)
```bash
cd ~/workspace
git clone https://github.com/<you>/heart-disease-mlops-pipeline.git && cd heart-disease-mlops-pipeline
python3 -m venv .venv && source .venv/bin/activate     # proves "clean setup from requirements"
pip install -U pip && pip install -r requirements-dev.txt
```
If `git push` from the VM asks for a password, use a GitHub **Personal Access Token** as the password
(Settings > Developer settings > Tokens, `repo` scope), then `git config user.name/user.email`.

## 2. Data, EDA, training, tests (30 min)
```bash
python scripts/download_data.py          # if blocked: download processed.cleveland.data
                                         # on your Mac and pass --from-file
python -m src.eda                        # figures -> reports/figures/
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_training.ipynb   # optional, slow
python -m src.train                      # ~1-3 min. Prints the model table + selected model
pytest                                   # screenshot -> 10_pytest_pass.png
flake8 src tests scripts                 # must print nothing
```
**MLflow UI** (second terminal, venv active):
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000     # open http://localhost:5000 in the VM Chrome
```
Screenshots: experiment page with the 3 nested runs (`03_mlflow_runs.png`), compare view or one
run's metrics/artifacts/plots (`04_mlflow_compare.png`).

Commit the data and model (needed for the Docker image, and the brief asks for the cleaned dataset):
```bash
git add -A && git commit -m "Add data, EDA figures, trained model" && git push
```

## 3. CI/CD evidence (10 min)
* GitHub > Actions: open the green run -> `01_ci_run.png`; scroll to **Artifacts** -> `02_ci_artifacts.png`.
* **Failure demo** (for the video and `11_ci_failure_demo.png`):
  ```bash
  git checkout -b break-demo
  sed -i 's/assert m\["test_roc_auc"\] > 0.6/assert m["test_roc_auc"] > 1.5/' tests/test_model.py
  git commit -am "Deliberately failing test" && git push -u origin break-demo
  ```
  Open GitHub > Actions and the run for branch `break-demo`: it fails at the pytest step. Screenshot it,
  then `git checkout main && git branch -D break-demo` (and delete the remote branch).

## 4. Docker (10 min)
```bash
make docker-build                      # passes the exact scikit-learn version of your training env
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0
docker ps                              # shows healthy after ~30 s
bash scripts/smoke_test.sh             # /health, 2 predictions, 1 expected 422
docker logs heart-api | tail           # JSON request logs
```
Screenshot terminal with build success + smoke test output -> `05_docker_run_predict.png`.
Also open `http://localhost:8000/docs` (Swagger) for the video. Then `docker rm -f heart-api`.

## 5. Kubernetes + monitoring (30 min)
```bash
bash scripts/deploy_k8s.sh             # starts minikube if needed, builds image inside it, applies manifests
minikube kubectl -- get pods,svc       # 2 api pods Running + prometheus + grafana -> 06_k8s_pods_svc.png
```
Expose the API (pick one):
```bash
minikube tunnel                        # separate terminal, keep running; then:
minikube kubectl -- get svc heart-api  # EXTERNAL-IP appears
curl http://<EXTERNAL-IP>/health
# or without tunnel:
minikube service heart-api --url
```
`bash scripts/smoke_test.sh <URL>` -> screenshot (`07_api_swagger_k8s.png` - Swagger or curl output).

Generate traffic, then open the monitoring UIs:
```bash
python scripts/generate_traffic.py --url <API_URL> --count 300
minikube service prometheus --url      # Status > Targets: heart-api UP     -> 08_prometheus_targets.png
minikube service grafana --url         # admin / admin -> "Heart Disease API Monitoring" -> 09_grafana_dashboard.png
```
Re-run the traffic script while the dashboard is open so the graphs have data.

Fallback if minikube misbehaves: `make compose-up` (API :8000, Prometheus :9090, Grafana :3000).
Say so honestly in the report, and still show the k8s manifests.

## 6. Final push and handover (5 min)
```bash
git status && git add -A && git commit -m "Final results" && git push
```
Pull on your Mac: `git pull`.

## 7. Report (Word)
On the Mac, with the screenshots in `screenshots/`:
```bash
pip install python-docx
python report/build_report.py --name "Your Name" --id "<ID>" --repo-url https://github.com/<you>/heart-disease-mlops-pipeline
```
Open `report/MLOps_Assignment1_Report.docx`, replace every **[WRITE YOUR OWN ...]** paragraph with
your own observations (this is what the integrity note rewards), check there are no `[MISSING IMAGE]`
placeholders, and make sure it is ~10 pages. Commit it to the repo too.

## 8. Screenshot checklist (exact names, in `screenshots/`)
| File | Content |
|---|---|
| 01_ci_run.png | green GitHub Actions run |
| 02_ci_artifacts.png | artifacts section of that run |
| 03_mlflow_runs.png | MLflow experiment, 3 model runs |
| 04_mlflow_compare.png | metrics / compare / plots |
| 05_docker_run_predict.png | docker build + run + sample prediction |
| 06_k8s_pods_svc.png | `get pods,svc` |
| 07_api_swagger_k8s.png | prediction through the K8s service |
| 08_prometheus_targets.png | target UP |
| 09_grafana_dashboard.png | dashboard with data |
| 10_pytest_pass.png | pytest all green |
| 11_ci_failure_demo.png | red run from the deliberate failure |

## 9. Submission checklist
- [ ] GitHub repo URL (public / access shared) with code, Dockerfile, requirements, cleaned data, notebooks, tests/, workflow YAML, k8s manifests, screenshots/, report
- [ ] Word report (~10 pages) uploaded to the portal
- [ ] Video (5-8 min) - see `docs/VIDEO_SCRIPT.md`
- [ ] Access instructions for the API (local URL + curl in README) since the VM URL is not public
- [ ] Submitted before 23:59 on 20 Oct 2026

## Troubleshooting
| Symptom | Fix |
|---|---|
| `docker: permission denied` | `sudo usermod -aG docker $USER` then `newgrp docker` (or prefix `sudo`) |
| Image pull error in minikube | check network; `docker pull python:3.12-slim` / `grafana/grafana:11.2.0` / `prom/prometheus:v2.54.1` on the VM first |
| Pod `ErrImageNeverPull`/`ImagePullBackOff` | the image must be built after `eval $(minikube docker-env)` (the script does this) |
| Pickle/sklearn version warning in container | rebuild with `make docker-build` (it pins the training version) |
| `minikube start` fails on resources | `minikube start --driver=docker --cpus=2 --memory=3072` |
| UCI download blocked | download on Mac, copy file into VM (paste via web-RDP clipboard / GitHub), use `--from-file` |
| CI failing on lint | run `flake8 src tests scripts` locally and fix before pushing |
