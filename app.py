"""
Smart Health Triage Assistant  (AI for Social Good + Responsible AI)
Experiment 11 (PBL-2) - PEIT21P Artificial Intelligence Lab

AI components (mapped to syllabus)
----------------------------------
1. Intelligent Agent   : Utility-based triage agent described with PEAS   (Module 2)
2. Search              : A* (vs BFS) route to the best hospital           (Module 3)
3. Knowledge/Reasoning : Rule base + forward chaining for danger signs    (Module 4)
4. Learning            : Decision-tree classifier for risk prediction     (Module 5)
5. Ethics              : explainable output, no identifiers, disclaimer,
                         fairness (age-group) and safety (under-triage) checks (Module 6)

NOTE: The patient data is SYNTHETIC (generated in this file). This is a student
prototype for decision support only - it does not diagnose any disease.

Run (demo)        : python smart_health_triage.py
Run (interactive) : python smart_health_triage.py --interactive
Output            : console report + triage_results.png
"""
import argparse
import heapq
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 7
random.seed(SEED)
rng = np.random.default_rng(SEED)

FEATURES = ["age", "heart_rate", "spo2", "temp_c", "systolic_bp",
            "chest_pain", "breathless", "cough", "headache"]
LEVELS = ["LOW", "MODERATE", "HIGH", "EMERGENCY"]


# ----------------------------------------------------------------------
# 1. SYNTHETIC DATA + MACHINE LEARNING  (Module 5)
# ----------------------------------------------------------------------
def risk_score(p):
    """Hidden 'ground truth' used only to create synthetic labels."""
    s = 0.0
    s += 3 if p["spo2"] < 92 else 1.5 if p["spo2"] < 95 else 0
    s += 1.5 if p["heart_rate"] > 110 else 0.5 if p["heart_rate"] > 100 else 0
    s += 1.5 if p["temp_c"] >= 39 else 0.8 if p["temp_c"] >= 38 else 0
    s += 2 if (p["systolic_bp"] >= 180 or p["systolic_bp"] < 90) else 0
    s += 2 * p["chest_pain"] + 1.5 * p["breathless"] + 0.3 * p["cough"] + 0.3 * p["headache"]
    s += 0.8 if (p["age"] >= 60 or p["age"] <= 5) else 0
    return s


def make_dataset(n=2000):
    X, y = [], []
    for _ in range(n):
        p = {
            "age": int(rng.integers(1, 91)),
            "heart_rate": float(np.clip(rng.normal(82, 18), 45, 160)),
            "spo2": float(np.clip(rng.normal(96.5, 2.5), 80, 100)),
            "temp_c": float(np.clip(rng.normal(37.2, 0.8), 35.5, 41)),
            "systolic_bp": float(np.clip(rng.normal(122, 22), 70, 210)),
            "chest_pain": int(rng.random() < 0.12),
            "breathless": int(rng.random() < 0.18),
            "cough": int(rng.random() < 0.30),
            "headache": int(rng.random() < 0.25),
        }
        s = risk_score(p) + rng.normal(0, 0.6)
        label = 0 if s < 1.5 else 1 if s < 3.5 else 2
        X.append([p[f] for f in FEATURES])
        y.append(label)
    return np.array(X), np.array(y)


X, y = make_dataset()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
model = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(X_tr, y_tr)


def age_group(age):
    return "Child (<=12)" if age <= 12 else "Elderly (>=60)" if age >= 60 else "Adult (13-59)"


# ----------------------------------------------------------------------
# 2. KNOWLEDGE BASE + FORWARD CHAINING  (Module 4)
# ----------------------------------------------------------------------
RULES = [
    # (id, required facts, concluded fact, explanation)
    ("R1", {"spo2<92", "breathless"}, "lvl_emergency", "very low oxygen with breathlessness"),
    ("R2", {"chest_pain", "tachycardia"}, "lvl_emergency", "chest pain with fast heart rate"),
    ("R3", {"chest_pain", "hypotension"}, "lvl_emergency", "chest pain with very low BP"),
    ("R4", {"bp_crisis", "headache"}, "lvl_emergency", "dangerously high BP with headache"),
    ("R5", {"bp_crisis", "chest_pain"}, "lvl_emergency", "dangerously high BP with chest pain"),
    ("R6", {"spo2<92"}, "lvl_high", "very low oxygen level"),
    ("R7", {"chest_pain"}, "lvl_high", "chest pain reported"),
    ("R8", {"high_fever", "vulnerable_age"}, "lvl_high", "high fever in child/elderly"),
    ("R9", {"hypotension"}, "lvl_high", "very low blood pressure"),
    ("R10", {"fever", "cough"}, "lvl_moderate", "fever with cough (possible infection)"),
    ("R11", {"spo2_mild"}, "lvl_moderate", "slightly low oxygen level"),
    ("R12", {"tachycardia"}, "lvl_moderate", "fast heart rate"),
    ("R13", {"fever"}, "lvl_moderate", "fever present"),
    ("R14", {"breathless"}, "lvl_moderate", "breathlessness reported"),
]


def extract_facts(p):
    f = set()
    if p["spo2"] < 92: f.add("spo2<92")
    elif p["spo2"] < 95: f.add("spo2_mild")
    if p["heart_rate"] > 110: f.add("tachycardia")
    if p["temp_c"] >= 39: f.add("high_fever")
    if p["temp_c"] >= 38: f.add("fever")
    if p["systolic_bp"] >= 180: f.add("bp_crisis")
    if p["systolic_bp"] < 90: f.add("hypotension")
    if p["age"] >= 60 or p["age"] <= 5: f.add("vulnerable_age")
    for s in ("chest_pain", "breathless", "cough", "headache"):
        if p[s]: f.add(s)
    return f


def forward_chain(p):
    facts = extract_facts(p)
    fired, changed = [], True
    while changed:                                  # repeat until nothing new is inferred
        changed = False
        for rid, cond, concl, why in RULES:
            if cond <= facts and concl not in facts:
                facts.add(concl)
                fired.append(f"{rid}: {why}")
                changed = True
    for lvl, name in (("lvl_emergency", 3), ("lvl_high", 2), ("lvl_moderate", 1)):
        if lvl in facts:
            return name, fired
    return 0, fired


# ----------------------------------------------------------------------
# 3. SEARCH: A* vs BFS on a city grid  (Module 3)
# ----------------------------------------------------------------------
N = 16
ROAD, TRAFFIC, CLOSED = 0, 1, 2
COST = {ROAD: 1, TRAFFIC: 3}
city = [[ROAD] * N for _ in range(N)]
for r in range(N):
    for c in range(N):
        x = random.random()
        city[r][c] = CLOSED if x < 0.10 else TRAFFIC if x < 0.28 else ROAD
hospitals = {"City Hospital": (2, 12), "Metro Care": (13, 3), "Sunrise Clinic": (12, 13)}
for pos in hospitals.values():
    city[pos[0]][pos[1]] = ROAD


def neighbours(pos):
    r, c = pos
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N and city[nr][nc] != CLOSED:
            yield (nr, nc)


def h(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])      # admissible heuristic


def astar(start, goal):
    pq, g, parent, expanded = [(h(start, goal), 0, start)], {start: 0}, {start: None}, 0
    while pq:
        _, cost, cur = heapq.heappop(pq)
        if cost > g[cur]:
            continue
        expanded += 1
        if cur == goal:
            path = []
            while cur:
                path.append(cur); cur = parent[cur]
            return path[::-1], cost, expanded
        for nxt in neighbours(cur):
            nc = cost + COST[city[nxt[0]][nxt[1]]]
            if nc < g.get(nxt, 1e9):
                g[nxt], parent[nxt] = nc, cur
                heapq.heappush(pq, (nc + h(nxt, goal), nc, nxt))
    return None, float("inf"), expanded


def bfs_nodes(start, goal):
    q, seen, n = deque([start]), {start}, 0
    while q:
        cur = q.popleft(); n += 1
        if cur == goal:
            return n
        for nxt in neighbours(cur):
            if nxt not in seen:
                seen.add(nxt); q.append(nxt)
    return n


def best_hospital(loc):
    best = None
    for name, pos in hospitals.items():
        path, cost, exp = astar(loc, pos)
        if path and (best is None or cost < best[2]):
            best = (name, path, cost, exp, bfs_nodes(loc, pos))
    return best


# ----------------------------------------------------------------------
# 4. UTILITY-BASED TRIAGE AGENT  (Module 2)
# ----------------------------------------------------------------------
ADVICE = {
    0: "Home care: rest, fluids, monitor symptoms. See a doctor if they worsen.",
    1: "Book a doctor / tele-consultation within 24-48 hours.",
    2: "See a doctor TODAY. Go to the nearest hospital; do not travel alone.",
    3: "EMERGENCY: call an ambulance now and go to the nearest hospital immediately.",
}


def triage(p, location=None):
    """Combines rule-based and ML assessments. Safety-first: takes the HIGHER level."""
    rule_lvl, fired = forward_chain(p)
    x = np.array([[p[f] for f in FEATURES]])
    proba = model.predict_proba(x)[0]
    ml_lvl = int(model.predict(x)[0])
    final = max(rule_lvl, ml_lvl)               # utility: missing a serious case costs more
    res = {"rule": rule_lvl, "ml": ml_lvl, "ml_conf": float(proba.max()),
           "final": final, "fired": fired, "advice": ADVICE[final], "route": None}
    if final >= 2 and location is not None:
        res["route"] = best_hospital(location)
    return res


# ----------------------------------------------------------------------
# 5. DEMO CASES, REPORT, FIGURES
# ----------------------------------------------------------------------
def case(id_, age, hr, spo2, t, sbp, cp=0, br=0, co=0, he=0):
    return {"id": id_, "age": age, "heart_rate": hr, "spo2": spo2, "temp_c": t,
            "systolic_bp": sbp, "chest_pain": cp, "breathless": br, "cough": co, "headache": he}


def free_cell():
    while True:
        r, c = random.randrange(N), random.randrange(N)
        if city[r][c] != CLOSED and (r, c) not in hospitals.values():
            return (r, c)


DEMO = [
    case("P01", 25, 76, 98, 36.8, 118),
    case("P02", 34, 92, 96, 38.4, 124, co=1),
    case("P03", 70, 105, 93, 39.2, 130, co=1),
    case("P04", 58, 118, 95, 37.0, 126, cp=1),
    case("P05", 66, 112, 89, 38.6, 85, br=1, co=1),
    case("P06", 52, 96, 97, 37.1, 190, he=1),
]


def read_patient():
    def ask(msg, cast=float):
        return cast(input(msg))
    p = {"id": "YOU"}
    p["age"] = ask("Age (years): ", int)
    p["heart_rate"] = ask("Heart rate (bpm): ")
    p["spo2"] = ask("SpO2 (%): ")
    p["temp_c"] = ask("Body temperature (deg C): ")
    p["systolic_bp"] = ask("Systolic BP (mmHg): ")
    for s in ("chest_pain", "breathless", "cough", "headache"):
        p[s] = 1 if input(f"{s.replace('_', ' ').title()}? (y/n): ").lower().startswith("y") else 0
    return p


def print_result(p, res):
    print(f"\n[{p['id']}] age {p['age']} | HR {p['heart_rate']:.0f} | SpO2 {p['spo2']:.0f}% | "
          f"Temp {p['temp_c']:.1f} | SBP {p['systolic_bp']:.0f}")
    print(f"  Rule-based: {LEVELS[res['rule']]:9} | ML: {LEVELS[res['ml']]:9} "
          f"(confidence {res['ml_conf']:.0%}) | FINAL: {LEVELS[res['final']]}")
    print("  Why (rules fired): " + ("; ".join(res["fired"]) if res["fired"] else "no danger-sign rule fired"))
    print("  Advice: " + res["advice"])
    if res["route"]:
        name, _, cost, exp, bfs_n = res["route"]
        print(f"  Route: {name} (travel cost {cost}; A* expanded {exp} nodes vs BFS {bfs_n})")
    print("  (Decision support only - not a medical diagnosis.)")


def evaluate():
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    print("-- Model evaluation (synthetic test set, 25%) --")
    print(f"Accuracy: {acc:.1%}")
    print("Confusion matrix (rows = actual, cols = predicted; LOW/MODERATE/HIGH):")
    print(cm)
    # safety: under-triage = actual HIGH predicted LOW
    high = y_te == 2
    under = int(np.sum((y_te == 2) & (pred == 0)))
    print(f"Dangerous misses (actual HIGH predicted LOW): {under} of {int(high.sum())}")
    print("\n-- Fairness check: accuracy by age group --")
    ages = X_te[:, 0]
    groups = np.array([age_group(a) for a in ages])
    for g in ("Child (<=12)", "Adult (13-59)", "Elderly (>=60)"):
        m = groups == g
        print(f"{g:15} n={int(m.sum()):3}  accuracy={accuracy_score(y_te[m], pred[m]):.1%}")
    return acc, cm


def make_figure(demo_results, loc_for_map):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    # (a) feature importance
    imp = model.feature_importances_
    order = np.argsort(imp)
    ax[0].barh([FEATURES[i] for i in order], imp[order], color="#2a7ab0")
    ax[0].set_title("What the ML model relies on\n(feature importance)")
    # (b) confusion matrix
    cm = confusion_matrix(y_te, model.predict(X_te))
    ax[1].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax[1].text(j, i, cm[i, j], ha="center", va="center")
    ax[1].set_xticks(range(3)); ax[1].set_yticks(range(3))
    ax[1].set_xticklabels(LEVELS[:3]); ax[1].set_yticklabels(LEVELS[:3])
    ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual"); ax[1].set_title("Confusion matrix")
    # (c) route map
    ax[2].imshow(city, cmap=ListedColormap(["#e8eef2", "#f3c26b", "#444444"]), vmin=0, vmax=2)
    for name, pos in hospitals.items():
        ax[2].scatter(pos[1], pos[0], marker="P", s=160, color="red", edgecolor="k", zorder=5)
        ax[2].text(pos[1] + .4, pos[0] - .5, name, fontsize=7)
    cols = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
    k = 0
    for p, res in demo_results:
        if res["route"]:
            path = res["route"][1]
            ax[2].plot([q[1] for q in path], [q[0] for q in path], lw=1.8, color=cols[k % 4])
            ax[2].scatter(path[0][1], path[0][0], s=60, color=cols[k % 4], edgecolor="k", zorder=6)
            ax[2].text(path[0][1] + .3, path[0][0] + .6, p["id"], fontsize=8)
            k += 1
    ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[2].set_title("A* routes to hospital (yellow = traffic, dark = closed)")
    plt.tight_layout()
    plt.savefig("triage_results.png", dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(" SMART HEALTH TRIAGE ASSISTANT")
    print("=" * 70)
    print("PEAS: Performance = correct urgency level, no dangerous misses | "
          "Environment = patient + city roads\n      Actuators = risk level, advice, hospital"
          " route | Sensors = symptoms, vital signs\n")
    evaluate()

    print("\n-- Triage of demo patients --")
    results = []
    for p in DEMO:
        loc = free_cell()
        res = triage(p, loc)
        results.append((p, res))
        print_result(p, res)

    if args.interactive:
        print("\n== Interactive triage ==")
        p = read_patient()
        res = triage(p, free_cell())
        print_result(p, res)
        results.append((p, res))

    make_figure(results, None)
    print("\nFigure saved as triage_results.png")


if __name__ == "__main__":
    main()