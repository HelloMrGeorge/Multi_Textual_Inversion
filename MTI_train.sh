export MODEL_NAME="/root/annotated_textual_inversion/model/sd-v1-5"

export DATA_DIR="./image"
export ANNO_PATH="./anno/CL4P.json"
export TOKEN="<CL4P>"
export INIT="robot"
export VALPROMPT="a photo of {} robot"
export PROPERTY="object"

rm -rf /root/annotated_textual_inversion/output/*

accelerate launch multi_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --train_annotation_path=$ANNO_PATH \
  --placeholder_token="$TOKEN" \
  --initializer_token="$INIT" \
  --validation_prompt="$VALPROMPT" \
  --learnable_property="$PROPERTY" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=3e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="output" --logging_dir="runs" \
  --mixed_precision="fp16" \
  --only_save_embeds \
  --checkpointing_steps=5000 --max_train_steps=1200 --save_steps=300 --validation_steps=300 \
  --embeddings_number=2