"""
File: dynamic-of-2d-critical-state-field.py
Author: Chuncheng Zhang
Date: 2024-04-24
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Simulation of the dynamic of the 2d critical state field 

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-24 ------------------------
# Requirements and constants
import cv2
import time
import contextlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from threading import Thread

from loguru import logger
from tqdm.auto import tqdm


# %% ---- 2024-04-24 ------------------------
# Constants

class Simulation:
    size = (200, 200)


class BoltzmannDistribution(object):
    k = 1e0
    temperature = 1e0
    cache = {}

    def clear_cache(self):
        # Clear the cache if k or temperature is changed
        self.cache = {}
        logger.debug('Cleared cache')

    def prob(self, energy: float):
        # Using the cache
        prob = self.cache.get(energy)
        if prob is not None:
            return prob

        # Computation is required
        den = self.k * self.temperature
        prob = np.exp(- (energy+1e-5) / den)
        self.cache[energy] = prob
        logger.debug(
            f'Recompute probability of energy: {energy} | prob: {prob}')

        return prob


sim = Simulation()
bzd = BoltzmannDistribution()

# %% ---- 2024-04-24 ------------------------
# Function and class


class HowFastIsIt(object):
    times = []

    @contextlib.contextmanager
    def timeit(self):
        try:
            tic = time.process_time()
            yield
        finally:
            t = time.process_time() - tic
            self.times.append(t)

    def report(self):
        n = len(self.times)
        mean = np.mean(self.times)
        std = np.std(self.times)
        logger.debug(
            f'Report the time cost with {n} samples, mean={mean}, std={std}')


class DynamicCriticalField(object):
    size = sim.size
    field = np.zeros(size)

    def convert_to_cv2(self):
        mat = np.concatenate([
            self.field[:, :, np.newaxis],
            self.field[:, :, np.newaxis],
            self.field[:, :, np.newaxis]
        ], axis=2)

        mat *= 255
        mat = mat.astype(np.uint8)

        # mat = cv2.resize(mat, (500, 500), interpolation=cv2.INTER_AREA)
        mat = cv2.resize(mat, (500, 500))

        return mat

    def compute_energy(self):
        field = self.field.copy()

        # --------------------
        random = np.random.random(field.shape)
        # field[random < 0.001] = 1

        # --------------------
        # Change energy change
        a = field[1:-1, 1:-1]
        _energy_change = a * 0

        b = field[:-2, 1:-1]
        _energy_change += 4 * a * b + 1 - 2 * a - 2 * b
        b = field[2:, 1:-1]
        _energy_change += 4 * a * b + 1 - 2 * a - 2 * b
        b = field[1:-1, :-2]
        _energy_change += 4 * a * b + 1 - 2 * a - 2 * b
        b = field[1:-1, 2:]
        _energy_change += 4 * a * b + 1 - 2 * a - 2 * b

        energy_change = field * 0
        energy_change[1:-1, 1:-1] = _energy_change

        # --------------------
        # Automatically downward the energy
        field[energy_change < 0] = 1 - field[energy_change < 0]

        # --------------------
        # Larger energy changes are changed by probability
        energies = [1, 2, 3, 4]
        probs = np.array([bzd.prob(e) for e in energies])

        for i, e in enumerate(energies):
            flip_prob = probs[i]
            flip_map = (energy_change == e) & (random < flip_prob)
            field[flip_map] = 1 - field[flip_map]

        self.field = field

        return field


# %% ---- 2024-04-24 ------------------------
# Play ground
if __name__ == "__main__":
    dcf = DynamicCriticalField()

    hfii = HowFastIsIt()

    # sns.heatmap(dcf.field)
    # plt.show()
    winname = 'main'
    total = int(1e4)
    frame_gap = 50  # milliseconds

    def loop():
        for j in range(total):
            with hfii.timeit():
                dcf.compute_energy()
            cv2.setWindowTitle(
                winname, f'{winname}: {j} | {total} |  {bzd.temperature} |')
            cv2.imshow(winname, dcf.convert_to_cv2())
            cv2.waitKey(frame_gap)

            if j % 100 == 0:
                hfii.report()

    Thread(target=loop, daemon=True).start()

    inp = ''
    while True:
        inp = input()

        if inp == 'q':
            break

        if inp.startswith('t'):
            bzd.temperature = float(inp[1:])
            bzd.clear_cache()

    print('Done.')


# %% ---- 2024-04-24 ------------------------
# Pending


# %% ---- 2024-04-24 ------------------------
# Pending
