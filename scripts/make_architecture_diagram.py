"""Draw docs/architecture.png (end-to-end MLOps architecture)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)


def arrow(x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color="#333"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, fontsize=8, ha="center")


ax.text(6.5, 6.7, "Heart Disease Risk - End-to-End MLOps Architecture",
        ha="center", fontsize=14, weight="bold")
# Row 1: data and training
box(0.3, 4.6, 2.0, 1.2, "UCI Heart Disease\ndataset\n(download script)", "#fde8c8")
box(3.0, 4.6, 2.2, 1.2, "Cleaning + EDA\nsrc/data.py\nsrc/eda.py", "#d6ecf7")
box(5.9, 4.6, 2.4, 1.2, "Feature pipeline +\nTraining (LR, RF, GB)\nGridSearchCV, 5-fold CV", "#d6ecf7")
box(9.0, 4.6, 1.8, 1.2, "MLflow\ntracking\n(params, metrics,\nplots, models)", "#e2d9f3")
box(11.2, 4.6, 1.6, 1.2, "Best pipeline\nmodels/*.joblib", "#d8f0d4")
arrow(2.3, 5.2, 3.0, 5.2)
arrow(5.2, 5.2, 5.9, 5.2)
arrow(8.3, 5.2, 9.0, 5.2)
arrow(10.8, 5.2, 11.2, 5.2)
# Row 2: CI/CD
box(0.3, 2.5, 2.0, 1.2, "Developer\ngit push", "#fde8c8")
box(3.0, 2.5, 3.6, 1.2, "GitHub Actions CI\nflake8 -> pytest -> train\n-> docker build + smoke test", "#f9d5d3")
box(7.3, 2.5, 2.2, 1.2, "Docker image\nFastAPI /predict\n/health /metrics", "#d8f0d4")
arrow(2.3, 3.1, 3.0, 3.1)
arrow(6.6, 3.1, 7.3, 3.1, "build")
arrow(11.9, 4.6, 8.8, 3.7, "packaged")
arrow(4.8, 3.7, 6.9, 4.6, "train in CI")
# Row 3: client, deployment, monitoring
box(0.3, 0.3, 2.0, 1.3, "Client\ncurl / Swagger UI", "#fde8c8")
box(3.0, 0.3, 4.2, 1.3, "Kubernetes (Minikube)\nDeployment x2 replicas + Service (LoadBalancer)\nreadiness / liveness probes", "#d6ecf7")
box(7.9, 0.3, 2.1, 1.3, "Prometheus\nscrapes /metrics", "#e2d9f3")
box(10.7, 0.3, 2.0, 1.3, "Grafana\ndashboard", "#e2d9f3")
arrow(2.3, 0.95, 3.0, 0.95, "POST /predict")
arrow(7.2, 0.95, 7.9, 0.95)
arrow(10.0, 0.95, 10.7, 0.95)
arrow(8.4, 2.5, 6.0, 1.6, "deploy")
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("saved", OUT)
