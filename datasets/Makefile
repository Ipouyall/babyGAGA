export HTTP_PROXY=10.2.32.28:10809
export HTTPS_PROXY=10.2.32.28:10809

babylm_10:
	python3 -m babylm --train --dataset ./text_data/train_10M --proc_dir ./processed/10M/train
	python3 -m babylm --test --dataset ./text_data/test --proc_dir ./processed/10M/test
	python3 -m babylm --evaluate --dataset ./text_data/dev --proc_dir ./processed/10M/dev

babylm_100:
	python3 -m babylm --train --dataset ./text_data/train_100M --proc_dir ./processed/100M/train
	python3 -m babylm --test --dataset ./text_data/test --proc_dir ./processed/100M/test
	python3 -m babylm --evaluate --dataset ./text_data/dev --proc_dir ./processed/100M/dev


all: babylm_10 babylm_100