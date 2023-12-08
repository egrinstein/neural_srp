.PHONY:

train:
	@python train_singlesource.py

train-multi:
	@python train_multisource.py

complexity:
	@python analyze_complexity.py

submit:
	@qsub qsub.pbs

viz:
	@python visualize_locata.py

viz-multi:
	@python visualize_tau.py