# ASD Step1 (fixed JSON + code)
This is a **Step 1** prototype: extract per-second emotion distributions from video (DeepFace),
extract coarse per-second audio-state distributions from wav (librosa heuristics),
combine with optional text priors, then match against prototype libraries using
DTW energy + Boltzmann/Gibbs fusion to produce a caregiver-facing narration.

## Files
- `run_step1.py` : end-to-end demo
- `train_calibrate.py` : calibrate fusion temperatures/weights on prototypes only
- `data/` : put your `demo.mp4` and JSON libs here (paths shown below)

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# copy your demo video and fixed json files
mkdir -p data
cp demo.mp4 data/demo.mp4
cp step1_vision_lib_fixed.json data/vision_lib.json
cp step1_audio_lib_fixed.json data/audio_lib.json
cp step1_text_priors_fixed.json data/text_priors.json
cp step1_narration_templates_fixed.json data/narr_templates.json

python train_calibrate.py --vision data/vision_lib.json --audio data/audio_lib.json --text data/text_priors.json --out data/calib.json
python run_step1.py --video data/demo.mp4 --text "school entrance, noisy hallway" --vision data/vision_lib.json --audio data/audio_lib.json --text_priors data/text_priors.json --narr data/narr_templates.json --calib data/calib.json
```
