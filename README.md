in aws g4dn.xlarge is enough.

all method has a common things as 
python run_something.py \
  --video data/demo.mp4 \
  --text "school entrance, noisy hallway" \
  --vision data/vision_lib.json \
  --audio data/audio_lib.json \
  --text_priors data/text_priors.json \
  --narr data/narr_templates.json \

Example:
python run_step1.py \
  --video data/demo.mp4 \
  --text "school entrance, noisy hallway" \
  --vision data/vision_lib.json \
  --audio data/audio_lib.json \
  --text_priors data/text_priors.json \
  --narr data/narr_templates.json \
  > outputs/step1_raw.txt

python rewrite_step1.py 
python rewrite_step1_human.py 

