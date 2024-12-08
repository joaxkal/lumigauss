#!/bin/bash
# LumiGauss: Training, Testing, and Rendering

##########################
# Dataset: ST
##########################
RESOLUTION=1

ST_SOURCE_PATH="path/to/st/dataset"
ST_OUTPUT_PATH="output/st"

mkdir -p "$ST_OUTPUT_PATH"

python train.py -s="$ST_SOURCE_PATH" -m="$ST_OUTPUT_PATH" --consistency_loss_lambda_init 10.0 -r $RESOLUTION --with_mlp

# Test with ground truth environment map
python test_gt_env_map.py -s="$ST_SOURCE_PATH" -m="$ST_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --test_config example_test_configs/st

# Render rotating environment map
python render_rotate_envmap.py -s="$ST_SOURCE_PATH" -m="$ST_OUTPUT_PATH" -r $RESOLUTION --with_mlp \
  --envmaps example_envmaps --viewpoints 12-04_18_00_DSC_0483.jpg 20-04_18_00_DSC_1473.jpg

# Render using an appearance list
python render_simple.py -s="$ST_SOURCE_PATH" -m="$ST_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --only_from_appearance_list --appearance_list example_test_configs/st/appearance_list.txt

##########################
# Dataset: LWP
##########################
RESOLUTION=1

LWP_SOURCE_PATH="path/to/lwp/dataset"
LWP_OUTPUT_PATH="output/lwp"

mkdir -p "$LWP_OUTPUT_PATH"

python train.py -s="$LWP_SOURCE_PATH" -m="$LWP_OUTPUT_PATH" --consistency_loss_lambda_init 10.0 -r $RESOLUTION --with_mlp

# Test with ground truth environment map
python test_gt_env_map.py -s="$LWP_SOURCE_PATH" -m="$LWP_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --test_config example_test_configs/lwp

# Render rotating environment map
python render_rotate_envmap.py -s="$LWP_SOURCE_PATH" -m="$LWP_OUTPUT_PATH" -r $RESOLUTION --with_mlp \
  --envmaps example_envmaps --viewpoints 26-04_17_50_DSC_2205.jpg 25-04_12_30_DSC_1976.jpg

# Render using an appearance list
python render_simple.py -s="$LWP_SOURCE_PATH" -m="$LWP_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --only_from_appearance_list --appearance_list example_test_configs/lwp/appearance_list.txt

##########################
# Dataset: LK2
##########################
RESOLUTION=1

LK2_SOURCE_PATH="path/to/lk2/dataset"
LK2_OUTPUT_PATH="output/lk2"

mkdir -p "$LK2_OUTPUT_PATH"

python train.py -s="$LK2_SOURCE_PATH" -m="$LK2_OUTPUT_PATH" --consistency_loss_lambda_init 10.0 -r $RESOLUTION --with_mlp

# Test with ground truth environment map
python test_gt_env_map.py -s="$LK2_SOURCE_PATH" -m="$LK2_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --test_config example_test_configs/lk2

# Render rotating environment map
python render_rotate_envmap.py -s="$LK2_SOURCE_PATH" -m="$LK2_OUTPUT_PATH" -r $RESOLUTION --with_mlp \
  --envmaps example_envmaps --viewpoints 12-04_10_00_DSC_0359.jpg 07-04_17_30_DSC_0089.jpg

# Render using an appearance list
python render_simple.py -s="$LK2_SOURCE_PATH" -m="$LK2_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --only_from_appearance_list --appearance_list example_test_configs/lk2/appearance_list.txt

##########################
# Dataset: TREVI
##########################
RESOLUTION=2

TREVI_SOURCE_PATH="path/to/trevi/dataset"
TREVI_OUTPUT_PATH="output/trevi"

mkdir -p "$TREVI_OUTPUT_PATH"

python train.py -s="$TREVI_SOURCE_PATH" -m="$TREVI_OUTPUT_PATH" --consistency_loss_lambda_init 1.0 -r $RESOLUTION \
  --with_mlp --iteration 60000 --start_shadowed 30500 --warmup 30000

# Render rotating environment map
python render_rotate_envmap.py -s="$TREVI_SOURCE_PATH" -m="$TREVI_OUTPUT_PATH" -r $RESOLUTION --with_mlp \
  --envmaps example_envmaps --viewpoints 38014238_3404678433.jpg 10011699_6167688545.jpg

# Render using an appearance list
python render_simple.py -s="$TREVI_SOURCE_PATH" -m="$TREVI_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --only_from_appearance_list --appearance_list example_test_configs/trevi/appearance_list.txt

