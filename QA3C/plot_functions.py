import os
import torch
import pickle
from collections import deque


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from os.path import abspath, dirname, join
## Ploting Function ##

def full_plotting(_fileTitle, _trainingLength, _currentRewardList):
	scores = []                        # list containing scores from each episode
	scores_std = []                    # List containing the std dev of the last 100 episodes
	scores_avg = []                    # List containing the mean of the last 100 episodes
	scores_window = deque(maxlen=100)  # last 100 scores

	reward_list = _currentRewardList
	for i_episode in range(_trainingLength):
		score = reward_list[i_episode]

		scores_window.append(score)       # save most recent score
		scores.append(score)              # save most recent score
		scores_std.append(np.std(scores_window)) # save most recent std dev
		scores_avg.append(np.mean(scores_window)) # save most recent std dev

	na_raw = np.array(scores)
	na_mu = np.array(scores_avg)
	na_sigma = np.array(scores_std)

	# plot the scores
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

	# plot the sores by episode
	ax1.plot(np.arange(len(na_raw)), na_raw, label='Raw Score')
	leg1 = ax1.legend();
	ax1.set_xlim(0, len(na_raw)+1)
	ax1.set_ylabel('Score')
	ax1.set_xlabel('Episode #')
	ax1.set_title('raw scores')

	# plot the average of these scores
	ax2.axhline(y=0.99, xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.7, alpha=0.9, label = 'Solved')
	ax2.plot(np.arange(len(na_mu)), na_mu, label='Average Score')
	leg2 = ax2.legend()
	ax2.fill_between(np.arange(len(na_mu)), na_mu+na_sigma, na_mu-na_sigma, facecolor='gray', alpha=0.1)
	ax2.set_ylabel('Average Score')
	ax2.set_xlabel('Episode #')
	ax2.set_title('average scores')

	f.tight_layout()

	# f.savefig(fig_prefix + 'dqn.eps', format='eps', dpi=1200)
	path = join(dirname(abspath(__file__)), _fileTitle + "_full.pdf")
	f.savefig(path,format = 'pdf')
	plt.close(f)