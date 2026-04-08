---
title: SecOps Alert Router
emoji: 🚨
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
---

# SecOps Alert Router V1

**Cybersecurity Incident Triage RL Environment | Meta PyTorch OpenEnv Hackathon**

An RL environment where AI agents learn to triage security alerts in a simulated Security Operations Center (SOC), balancing investigation speed (MTTC) against accuracy (FPR).

## Problem

SOCs face 10,000+ alerts/day. Analysts suffer alert fatigue, miss critical threats, and respond slowly. Static SIEM playbooks can't adapt. RL agents can learn the optimal investigate-vs-contain tradeoff dynamically.

## Environment

- **Observation**: alert type (5 categories), severity (3 levels), confidence score, evidence flags
- **Actions**: 8 discrete — 4 safe (investigate), 3 risky (contain), 1 resolve
- **Partial observability**: agent cannot see ground truth (`is_true_threat`) — must infer from evidence
- **Episodes**: 1-5 steps, terminates on containment, resolution, or timeout

### Reward Function

| Condition | Reward |
|-----------|--------|
| True Positive (contain real threat) | +10 |
| Fast critical containment (≤3 steps) | +12 |
| False Positive (contain benign) | -10 |
| True Negative (resolve benign) | +1 |
| False Negative (miss threat) | -50 |
| Investigation step | -1 |
| Procedure violation (contain without evidence) | -5 |

## Tasks

| Task | Difficulty | Episodes | Focus |
|------|-----------|----------|-------|
| `benign-filter` | Easy | 5 | Identify and resolve benign alerts |
| `mixed-triage` | Medium | 10 | Balance speed vs accuracy across mixed alerts |
| `critical-escalation` | Hard | 10 | Rapidly contain critical threats under pressure |

## Quick Start

```bash
pip install -r requirements.txt
python inference.py
```

### Run Tests

```bash
python tests/test_environment.py
```

### Start Server

```bash
cd secops_env
python -m secops_env.server.app
# Server at http://localhost:8000
```

### Docker

```bash
docker build -t secops-env -f secops_env/server/Dockerfile .
docker run -p 8000:8000 secops-env
```

## Usage

```python
from secops_env.models import SecOpsAction
from secops_env.server.secops_environment import SecOpsEnvironment

env = SecOpsEnvironment(task_name="mixed-triage", seed=42)
obs = env.reset()

obs = env.step(SecOpsAction(action_id=1))  # query_logs
print(f"Confidence: {obs.confidence_score}")

obs = env.step(SecOpsAction(action_id=5))  # isolate_host
print(f"Reward: {obs.reward}, Done: {obs.done}")
```

## Project Structure

```
├── inference.py              # Hackathon entry point (LLM + heuristic fallback)
├── requirements.txt          # Dependencies
├── tests/
│   └── test_environment.py   # 7 programmatic tests
└── secops_env/
    ├── models.py             # Pydantic models (Action, Observation, State)
    ├── client.py             # WebSocket client
    ├── openenv.yaml          # OpenEnv manifest
    ├── pyproject.toml        # Package config
    └── server/
        ├── app.py            # FastAPI server
        ├── secops_environment.py  # Core environment logic
        ├── alert_generator.py     # Synthetic alert factory
        ├── tasks.py          # 3 task configs + graders
        ├── rubrics.py        # Trajectory rubric
        └── Dockerfile        # Container deployment
```

## Baseline Scores (Heuristic Policy)

| Task | Score | Correct |
|------|-------|---------|
| benign-filter | 0.82 | 5/5 |
| mixed-triage | 0.69 | 8/10 |
| critical-escalation | 0.82 | 9/10 |

## OpenEnv Compliance

Built on the OpenEnv framework with standard `step()` / `reset()` / `state()` API. Passes `openenv validate` (local + runtime).

## License

BSD 3-Clause License
