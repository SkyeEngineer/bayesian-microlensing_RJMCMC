"""Interface with neural network surrogate posteriors."""

import pickle
import numpy as np

# File access. 
import os
import os.path
from pathlib import Path

from sklearn.cluster import OPTICS
from sklearn.preprocessing import MinMaxScaler

class Surrogate_Posterior(object):
    def __init__(self, m, data):
        """Initialises the model."""
        self.m = m

        """Get a single or binary model posterior.
            
        Args:
            m: [int] Model index, single or binary, 0 or 1.

        Returns:
            posterior: [pickle] Posterior object.
        """

        path = os.getcwd()
        #path = (str(Path(path).parents[0]))

        if m == 0:
            with open(path+"/distributions/single_25K_720.pkl", "rb") as handle: distribution = pickle.load(handle)

        if m == 1:
            with open(path+"/distributions/binary_100K_720.pkl", "rb") as handle: distribution = pickle.load(handle)

        self.distribution = distribution

        self.data = data


    def sample(self, n):
        self.samples = self.distribution.sample((n,), x=self.data, show_progress_bars=False)
        
        return

    def get_modes(self, latex_output = False):
        """Get the modes of a multidimensional sampled distribution using the OPTICS sampler.

        Args:
            samples (np.ndarray): samples to find modes from.
            latex_output (bool, optional): if latex output of modes is wanted. Defaults to False.

        Returns:
            np.ndarray: array of mode centre locations.
        """

        # Fit min-max scaler to ensure each dimension is handled similarly
        scaled_samples = MinMaxScaler().fit_transform(self.samples.numpy())

        # Apply OPTICS sampler with specified settings and fit to samples
        clust = OPTICS(min_samples=50, min_cluster_size=100, xi=0.05, max_eps=0.1).fit(
            scaled_samples
        )

        labels = clust.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print(f"{n_clusters_} modes, {n_noise_} samples not assigned... ")

        modes = []

        # Go through each cluster and find statistics
        for i in range(n_clusters_):
            samples_i = self.samples[labels == i]

            mode_i = np.zeros((samples_i.shape[1]))

            print(f"\nMode: {i}")

            latex_string = ""

            for j in range(samples_i.shape[1]):
                # If the number of clusters is less than 2, just use all the samples
                if n_clusters_ < 2:
                    temp = np.percentile(self.samples[:, j], [16, 50, 84])
                else:
                    temp = np.percentile(samples_i[:, j], [16, 50, 84])

                print(f"{temp[1]:.4f} +{temp[2]-temp[1]:.4f} -{temp[1]-temp[0]:.4f}")

                latex_string += f"${temp[1]:.4f}^{{+{temp[2]-temp[1]:.4f}}}_{{-{temp[1]-temp[0]:.4f}}}$ & "

                mode_i[j] = temp[1]

            modes.append(mode_i)

            if latex_output:
                print(latex_string)

        self.modes = np.array(modes)

        return


    def max_aposteriori(self):
        """Maximise a posterior.
            
        The input signal_data conditions the posterior to data.

        Args:
            posterior: [pickle] Posterior object.
            signal_data: [list] Measured flux signals at discrete times.

        Returns:
            centre: [list] Estimated parameter values of maximum.
        """
        centre = np.array(np.float64(self.distribution.map(self.data, num_iter = 100, num_init_samples = 100, show_progress_bars = False)))
        
        print(centre)

        return centre






