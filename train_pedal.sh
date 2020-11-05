# $ nohup ./train_pedal.sh &

# ============ Train piano transcription system from scratch ============
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="../training_data/MAESTRO"

# Modify to your workspace
WORKSPACE="./models/uni_aug"

# --- 1. Train note transcription system ---
python3 pytorch/main.py train \
  --workspace=$WORKSPACE \
  --data_dir=$DATASET_DIR \
  --model_type='Regress_pedal_CRNN' \
  --loss_type='regress_pedal_bce' \
  --augmentation='aug' \
  --max_note_shift=0 \
  --batch_size=4 \
  --learning_rate=5e-4 \
  --reduce_iteration=10000 \
  --resume_iteration=-1 \
  --validation_iteration=5000 \
  --checkpoint_iteration=10000 \
  --early_stop=300000 \
  --unidirectional \
  --cuda
