"""Send random but realistic patient requests to the API to populate logs and dashboards.

    python scripts/generate_traffic.py --url http://localhost:8000 --count 200
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient(rng: random.Random) -> dict:
    age = rng.randint(30, 77)
    return {
        "age": age, "sex": rng.randint(0, 1), "cp": rng.randint(1, 4),
        "trestbps": rng.randint(95, 190), "chol": rng.randint(140, 400),
        "fbs": rng.randint(0, 1), "restecg": rng.randint(0, 2),
        "thalach": rng.randint(90, 200), "exang": rng.randint(0, 1),
        "oldpeak": round(rng.uniform(0, 4.5), 1), "slope": rng.randint(1, 3),
        "ca": rng.randint(0, 3), "thal": rng.choice([3, 6, 7]),
    }


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url + "/predict", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--invalid-every", type=int, default=25,
                    help="send an invalid request every N calls (shows 422s on the dashboard)")
    args = ap.parse_args()
    rng = random.Random(7)
    codes = {}
    for i in range(1, args.count + 1):
        payload = {"age": 999} if args.invalid_every and i % args.invalid_every == 0 \
            else random_patient(rng)
        code = post(args.url, payload)
        codes[code] = codes.get(code, 0) + 1
        time.sleep(args.delay)
    print("status codes:", codes)


if __name__ == "__main__":
    main()
