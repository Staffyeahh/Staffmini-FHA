@echo off
cd /d "%~dp0"
python scripts\smoke_test.py --config configs\model_config_v14_2_gated_delta_anchor_50m.yaml --seq-len 512 --batch-size 4
python scripts\train.py --model-config configs\model_config_v14_2_gated_delta_anchor_50m.yaml --train-config configs\train_config_300.yaml --token-bin data\tokenized\cosmopedia_mix_100m.bin --output-dir outputs\v14_2_gated_delta_300
python scripts\eval_causal_suite.py --model-config configs\model_config_v14_2_gated_delta_anchor_50m.yaml --checkpoint outputs\v14_2_gated_delta_300\step_0000300\model.pt --token-bin data\tokenized\cosmopedia_mix_5m.bin --sequence-lengths 512 1024 2048 --batches 50 --batch-size 8 --output-json outputs\v14_2_gated_delta_300\eval_causal_300.json --output-md outputs\v14_2_gated_delta_300\eval_causal_300.md
