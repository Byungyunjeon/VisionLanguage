# Interpreting Emotional Sequences in Autistic Children with Multimodal Temporal AI

Autistic children often express distress through **temporal patterns of behavior** rather than isolated facial expressions. Caregivers may misinterpret these reactions by focusing on visible context (e.g., “it’s noisy”) instead of the **latent cause** (e.g., loss of control, disrupted expectations).

This repository presents a **multimodal temporal AI framework** that interprets emotional behavior as a **time-dependent sequence**, integrating:
- per-second **facial emotion probability vectors** from video,
- per-second **audio arousal cues** from the same recording (no speech recognition),
- short **prototype trajectories** representing caregiver-relevant latent functions.

Observed behavior is aligned to these prototypes using temporal alignment (DTW). An **energy-based, uncertainty-preserving inference scheme** converts alignment energy into probabilistic interpretations that remain cautious and interpretable. The system generates **parent-facing explanations** that help translate observed behavior into actionable understanding.

> Research prototype only. Not a diagnostic or medical system.

---

## Key idea

Instead of predicting a single emotion label, the system infers a **latent caregiver-relevant function** (e.g., `repair_integrity`, `share_control`) that best explains the **temporal evolution** of multimodal signals.

Uncertainty is preserved explicitly so that explanations remain cautious rather than overconfident.

---

## Repository overview

- `run_step1.py`  
  Full Step-1 pipeline (vision + audio + text priors + optional narration + optional calibration)

- `run_step1_baseline0.py`  
  Minimal baseline (vision only, no text priors, no physics-style weighting)

- `run_step1_baseline1.py`  
  Sequence-aware baseline with text priors but **without** physics-style Boltzmann weighting

- `train_calibrate_fixed.py`  
  Learns fusion temperatures and weights from prototype libraries and text priors

- `rewrite_step1*.py`  
  Deterministic rewrite of raw output into a caregiver-facing explanation

- `rewrite_*_with_ollama.py`  
  Optional LLM-based rewriting using Ollama (local inference)

---

## Environment requirements

### Python
- Python **3.10+**

### Core Python packages
(see `requirements.txt`)

- numpy, scipy
- opencv-python
- deepface (+ retina-face, mtcnn)
- tensorflow (CPU or GPU)
- librosa (audio)
- torch (calibration)
- tqdm, pillow, pyyaml

### Optional (LLM rewrite)
- **Ollama** system binary
- Model: `qwen2.5:7b-instruct`

Ollama is **not** a Python package and must be installed separately.

---

## AWS setup (example)

**Instance**: `g4dn.xlarge`  
- 1 × NVIDIA T4 (16GB)
- 4 vCPUs
- 16GB RAM
- ~100GB SSD

```bash
ssh -i "alscoregon2026.pem" ubuntu@ec2-35-91-112-78.us-west-2.compute.amazonaws.com

cd asd/asd1/
source .venv/bin/activate
````

To copy results back:

```bash
scp -i "alscoregon2026.pem" \
  ubuntu@ec2-35-91-112-78.us-west-2.compute.amazonaws.com:/home/ubuntu/asd/asd1/pyasd23jan.tar.gz .
```

---

## Running examples

### BASE0 (minimal sanity baseline)

```bash
python run_step1_baseline0.py \
  --video data/demo.mp4

python rewrite_step1_baseline0_human.py
python rewrite_baseline0_with_ollama.py
```

**Purpose**: fast check that vision extraction works; no text priors, no physics layer.

---

### BASE1 (sequence-aware, no physics layer)

```bash
python run_step1_baseline1.py \
  --video data/demo.mp4 \
  --text "school entrance, noisy hallway" \
  --vision data/vision_lib.json \
  --audio data/audio_lib.json \
  --text_priors data/text_priors.json

python rewrite_step1_baseline1_human.py
python rewrite_baseline1_with_ollama.py
```

**Purpose**: temporal alignment + text priors, but plain softmax over energies.

---

### Step-1 (full model, no calibration)

```bash
python run_step1.py \
  --video data/demo.mp4 \
  --text "school entrance, noisy hallway" \
  --vision data/vision_lib.json \
  --audio data/audio_lib.json \
  --text_priors data/text_priors.json \
  --narr data/narr_templates.json

python rewrite_step1.py
python rewrite_with_ollama.py
```

**Purpose**: full multimodal temporal inference with physics-style weighting.

---

### Step-1 with calibration (recommended)

```bash
python train_calibrate_fixed.py \
  --vision data/vision_lib.json \
  --audio data/audio_lib.json \
  --text_priors data/text_priors.json \
  --out data/calib.json
```

Then:

```bash
python run_step1.py \
  --video data/demo.mp4 \
  --text "school entrance, noisy hallway" \
  --vision data/vision_lib.json \
  --audio data/audio_lib.json \
  --text_priors data/text_priors.json \
  --narr data/narr_templates.json \
  --calib data/calib.json

python rewrite_step1.py
python rewrite_with_ollama.py
```

**Purpose**: adjust temperatures and fusion weights so uncertainty is communicated more cautiously.

---

## Baseline comparison (conceptual)

| Mode   | Temporal | Text prior | Physics weighting | Calibration |
| ------ | -------- | ---------- | ----------------- | ----------- |
| BASE0  | ✗        | ✗          | ✗                 | ✗           |
| BASE1  | ✓        | ✓          | ✗                 | ✗           |
| Step-1 | ✓        | ✓          | ✓                 | optional    |

---

## Output

Step-1 produces:

* ranked latent functions with probabilities,
* uncertainty indicators (e.g., risk / flip likelihood),
* a caregiver-facing explanation explaining **why** the behavior may have occurred,
* optional LLM-refined narration.

---

## Notes

* Audio is used **only as arousal / state information** (no ASR).
* The system intentionally avoids hard classification and preserves uncertainty.
* Live progress and timestamps are printed for long-running steps.

---

## Citation

**Interpreting emotional sequences in autistic children with multimodal temporal AI**
Byungyun Jeon

---

## Physics-inspired inference and uncertainty

This system borrows concepts from **statistical physics** to reason about uncertainty in emotional interpretation. The goal is *not* to claim a physical model of the brain, but to use physics-inspired tools that naturally handle **soft evidence, competing explanations, and uncertainty**.

---

### Energy from temporal alignment (DTW)

Observed behavior is a **time series**
[
X = (x_1, x_2, \dots, x_T)
]
where each (x_t) is a multimodal state (facial emotion probabilities + audio arousal features).

Each latent function (f) (e.g., *repair integrity*, *share control*) is represented by a **prototype trajectory**:
[
P_f = (p_1, p_2, \dots, p_{L_f})
]

We compute an **alignment energy** using Dynamic Time Warping (DTW):
[
E_f = \mathrm{DTW}(X, P_f)
]

* Lower (E_f) means the observed behavior evolves similarly to the prototype.
* Higher (E_f) means poor temporal alignment.
* DTW allows stretching and compression in time, which is critical for real child behavior.

This energy plays the role of a **negative log-likelihood**: smaller energy → more plausible explanation.

---

### Gibbs / Boltzmann weighting over explanations

Instead of choosing the single lowest-energy explanation, we convert energies into a **probability distribution** using a Gibbs (Boltzmann) distribution:

[
P(f \mid X) =
\frac{\exp(-E_f / T)}{\sum_{f'} \exp(-E_{f'} / T)}
]

Where:

* (E_f) is the DTW alignment energy
* (T > 0) is a **temperature** parameter

#### Interpretation of temperature

* **Low temperature** ((T \downarrow)):

  * Strong preference for the minimum-energy explanation
  * Sharper, more confident predictions
  * Lower entropy

* **High temperature** ((T \uparrow)):

  * Energies are flattened
  * Multiple explanations receive non-negligible probability
  * Higher entropy → **greater acknowledged uncertainty**

This matches caregiver reality: when evidence is weak or ambiguous, the system should *not* sound confident.

---

### Entropy as a measure of uncertainty

The distribution (P(f \mid X)) has entropy:
[
H = -\sum_f P(f \mid X)\log P(f \mid X)
]

* High entropy → “many explanations plausible”
* Low entropy → “one explanation dominates”

Temperature calibration explicitly controls this behavior.

---

### Landau-style barrier for emotional instability

Some situations involve **state instability** rather than a stable latent cause (e.g., a child close to meltdown).

We model this using a **Landau-style potential** over an abstract emotional order parameter (m):

[
V(m) = a m^2 + b m^4
]

* Two shallow minima → competing emotional states
* A small perturbation can flip the system

From this, we estimate:

* **Barrier height** (how stable the current state is)
* **Flip probability** using a Glauber-type dynamics:
  [
  P_{\text{flip}} \approx \exp(-\Delta V / T)
  ]

Where:

* (\Delta V) is the energy barrier between states
* (T) again represents environmental or internal uncertainty

This provides a *qualitative* signal such as:

> “risk(meltdown-ish) ≈ 0.51, flip ≈ 0.65”

These values are **not diagnoses**, but warnings that the system is near an unstable region.

---

### Why this matters for caregiver-facing AI

* Physics-style inference avoids brittle hard labels
* Uncertainty is **explicit**, not hidden
* Explanations can say:

  > “Several explanations are plausible; this one is slightly more consistent”

This is ethically preferable to overconfident classification when interpreting vulnerable populations.

---

### Calibration as temperature learning

The calibration step learns:

* video temperature (T_v)
* audio temperature (T_a)
* fusion weights (\alpha_v, \alpha_a, \alpha_t)

by minimizing prediction loss over known prototype associations.

This aligns numerical confidence with observed reliability.

---

### Summary intuition

| Physics concept  | Interpretation in this system                      |
| ---------------- | -------------------------------------------------- |
| Energy           | Temporal mismatch between behavior and explanation |
| Temperature      | Willingness to admit uncertainty                   |
| Entropy          | How ambiguous the situation is                     |
| Barrier height   | Emotional stability                                |
| Flip probability | Risk of sudden state change                        |

---
