export MODEL_NAME="/root/annotated_textual_inversion/model/sd-v1-5"

# paper cutting

export DATA_DIR="/root/annotated_textual_inversion/data/cutting"
export ANNO_PATH="/root/annotated_textual_inversion/anno/cutting.json"
export TOKEN="<cutting>"
export INIT="style"
export VALPROMPT="a photo in the style of <cutting>"

# liukanshan

export DATA_DIR="/root/annotated_textual_inversion/data/liukanshan"
export ANNO_PATH="/root/annotated_textual_inversion/data/liukanshan.json"
export TOKEN="<liukanshan>"
export INIT="cartoon"
export VALPROMPT="a photo in the style of <liukanshan>"

# CL4P

export DATA_DIR="/root/annotated_textual_inversion/data/CL4P"
export ANNO_PATH="/root/annotated_textual_inversion/anno/CL4P.json"
export TOKEN="<CL4P>"
export INIT="robot"
export VALPROMPT="a photo of {} robot"
export PROPERTY="object"

# bird

export DATA_DIR="/root/annotated_textual_inversion/data/bird"
export ANNO_PATH="/root/annotated_textual_inversion/anno/bird.json"
export TOKEN="<bird>"
export INIT="bird"
export VALPROMPT="a photo of {} bird"
export PROPERTY="object"

# Strelitzia

export DATA_DIR="/root/annotated_textual_inversion/data/Strelitzia"
export ANNO_PATH="/root/annotated_textual_inversion/anno/Strelitzia.json"
export TOKEN="<flower>"
export INIT="flower"
export VALPROMPT="a photo of {} flower"
export PROPERTY="object"

# Monet

export DATA_DIR="/root/annotated_textual_inversion/data/Monet"
export ANNO_PATH="/root/annotated_textual_inversion/anno/Monet.json"
export TOKEN="<Monet>"
export INIT="impressionism"
export VALPROMPT="a photo in the style of {}"
export PROPERTY="style"

# mocha

export DATA_DIR="/root/annotated_textual_inversion/data/mocha"
export TOKEN="<mocha>"
export INIT="style"
export VALPROMPT="a photo in the style of <mocha>"
