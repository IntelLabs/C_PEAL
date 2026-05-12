# -------------VIT BACKBONE RUNS
export CUDA_VISIBLE_DEVICES=5 # Multi GPU not supported yet for ALVLM trainer, so set single GPU
WORKSPACE=/home/athmanar/C_PEAL/C_PEAL/ # edit worspace
DATASET=dtd
EPOCH=10
OUTDATA=R50RES
MODELNM=rn50
##########################################################ENTROPY
METHOD=entropy
SEED=1
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=2
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=3
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
########################################################## RANDOM
METHOD=random
SEED=1
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=2
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=3
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
##########################################################ENTROPY=1.0+INTERW+SECOND+ANN (OURS)
METHOD=entropy
SECONDLOSSW=1.0
SEED=1
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --secondary_calib_loss --secondary_calib_anneal --secondary_calib_interWratio --secondary_calib_lossw ${SECONDLOSSW} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=2
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --secondary_calib_loss --secondary_calib_anneal --secondary_calib_interWratio --secondary_calib_lossw ${SECONDLOSSW} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=3
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --secondary_calib_loss --secondary_calib_anneal --secondary_calib_interWratio --secondary_calib_lossw ${SECONDLOSSW} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
##########################################################BADGE
METHOD=badge
SEED=1
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=2
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=3
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
##########################################################CORESET
METHOD=coreset
SEED=1
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=2
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1
SEED=3
python ${WORKSPACE}src/train.py --root ${WORKSPACE}DATA --seed ${SEED} --max-epochs $EPOCH --trainer ALVLM --dataset-config-file ${WORKSPACE}configs/datasets/${DATASET}.yaml --config-file ${WORKSPACE}configs/trainers/ALVLM/${MODELNM}.yaml --output-dir ${WORKSPACE}results/${OUTDATA}/${DATASET}/ALVLM/${MODELNM}_-1shots/nctx16_cscTrue_ctpend_al_${METHOD}_mode_none_E$EPOCH/seed${SEED} TRAINER.COOP.N_CTX 16 TRAINER.COOP.CSC True TRAINER.COOP.CLASS_TOKEN_POSITION end DATASET.NUM_SHOTS -1 TRAINER.COOPAL.METHOD ${METHOD} TRAINER.COOPAL.GAMMA 0.1


