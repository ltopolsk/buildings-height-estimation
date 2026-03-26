@echo off
echo =======================================================
echo BEGINNING OF TRAINING SESSION - SOLOv2
echo =======================================================

call C:\Users\ltopolsk\miniconda3\Scripts\activate.bat mgr

set PYTHONPATH=.

:: ---------------------------------------------------------
:: EARLY FUSION
:: ---------------------------------------------------------
echo [1/4] RUNNING: EARLY FUSION ...
mim train mmdet models_config/config_early.py --work-dir runs/early_fusion
if %errorlevel% neq 0 echo [ERROR] Early fusion training stopped! & goto :exp2
echo [SUCCESS] Early fusion model trained.

:exp2
:: ---------------------------------------------------------
:: FEATURE FUSION
:: ---------------------------------------------------------
echo [2/4] RUNNING: FEATURE FUSION ...
mim train mmdet models_config/config_feature.py --work-dir runs/feature_fusion
if %errorlevel% neq 0 echo [ERROR] Feature fusion training stopped! & goto :exp3
echo [SUCCESS] Feature fusion model trained.

:exp3
:: ---------------------------------------------------------
:: LATE FUSION (RGB)
:: ---------------------------------------------------------
echo [3/4] RUNNING: DECISION FUSION (RGB)...
mim train mmdet models_config/config_late_rgb.py --work-dir runs/late_fusion_rgb
if %errorlevel% neq 0 echo [ERROR] RGB model training stopped! & goto :exp4
echo [SUCCESS] RGB model trained.

:exp4
:: ---------------------------------------------------------
:: LATE FUSION (SAR)
:: ---------------------------------------------------------
echo [4/4] RUNNING: DECISION FUSION (SAR)...
mim train mmdet models_config/config_late_sar.py --work-dir runs/late_fusion_sar
if %errorlevel% neq 0 echo [ERROR] SAR model training stopped!
echo [SUCCESS] SAR model trained.

echo =======================================================
echo FINISHED TRAINING
echo =======================================================
pause