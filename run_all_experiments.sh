#!/bin/bash
echo "======================================================="
echo "BEGINNING OF TRAINING SESSION - SOLOv2"
echo "======================================================="

eval "$(conda shell.bash hook)"
conda activate mgr

export PYTHONPATH="."

# ---------------------------------------------------------
# 1: Early Fusion
# ---------------------------------------------------------
echo "[1/4] RUNNING: EARLY FUSION ..."
mim train mmdet models_config/config_early.py --work-dir runs/early_fusion
if [ $? -ne 0 ]; then 
    echo "[ERROR] Early fusion training stopped!"
else 
    echo "[SUCCES] Early fusion model trained."
fi

# ---------------------------------------------------------
# 2: Feature-Level
# ---------------------------------------------------------
echo "[2/4] RUNNING: FEATURE FUSION ..."
mim train mmdet models_config/config_feature.py --work-dir runs/feature_fusion
if [ $? -ne 0 ]; then 
    echo "[ERROR] Feature fusion training stopped!"
else 
    echo "[SUCCESS] Feature fusion model trained."
fi

# ---------------------------------------------------------
# 3: LATE FUSION (RGB)
# ---------------------------------------------------------
echo "[3/4] RUNNING: DECISION FUSION (RGB)..."
mim train mmdet models_config/config_late_rgb.py --work-dir runs/late_fusion_rgb
if [ $? -ne 0 ]; then 
    echo "[ERROR] RGB model training stopped!"
else 
    echo "[SUCCESS] RGB model trained."
fi

# ---------------------------------------------------------
# 4: LATE FUSION (SAR)
# ---------------------------------------------------------
echo "[4/4] RUNNING: DECISION FUSION (SAR)..."
mim train mmdet models_config/config_late_sar.py --work-dir runs/late_fusion_sar
if [ $? -ne 0 ]; then 
    echo "[ERROR] SAR model training stopped!"
else 
    echo "[SUCCES] SAR model trained."
fi

echo "======================================================="
echo "TRAINING FINISHED"
echo "======================================================="