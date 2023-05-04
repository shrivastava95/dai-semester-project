python3 eval.py \
       	--exp sample \
	--workers 2  \
	--epochs 10  \
	--start-epoch 0 \
	--batch-size 16 \
	--learning-rate 0.0003 \
	--weight-decay 0 \
	--save-freq 1 \
	--print-iter 1 \
	--save-dir noise  \
	--test 0 \
	--test_e4 0 \
	--defense 1 \
	--optimizer adam