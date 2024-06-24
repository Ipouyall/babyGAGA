export HTTP_PROXY=10.2.32.28:10809
export HTTPS_PROXY=10.2.32.28:10809

python3 -m babylm --train --dataset ./text_data/train_100M --proc_dir ./processed/100M/train
python3 -m babylm --test --dataset ./text_data/test --proc_dir ./processed/100M/test
python3 -m babylm --evaluate --dataset ./text_data/dev --proc_dir ./processed/100M/dev
