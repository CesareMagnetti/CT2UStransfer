# Experiment for CycleGAN and Constrastive-Unpaired-Transaltion (CUT) baselines

Given that repo for CUT and CycleGAN is very well made, we can use most of their code to train two baselines for our project, one using CycleGAN and one using CUT. Results should not be groundbreaking but at the very least they should work. These will be used as baselines for future experiments in my Thesis.

# Logs

-there is a bug in CUT repo for grayscale -> grayscale translation. In data/unaligned_dataset.py line 67 replace: ```transform = get_transform(modified_opt)``` with ```transform = get_transform(modified_opt, grayscale=True)```.
