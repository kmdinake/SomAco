#!usr/bin/bash
from sklearn.datasets import load_iris
from scipy.spatial.distance import euclidean
import numpy as np


class SOM(object):
    def __init__(self, epochs=100):
        self.X = None  # Number of Cols
        self.Y = None  # Number of Rows
        self.map = None
        self.epochs = epochs
        self.tau_1 = 0
        self.tau_2 = 0
        self.kernel_width = 0  # aka the neighborhood radius
        self.initial_kernel_width = 0
        self.learning_rate = 0
        self.initial_learning_rate = 0

    def train(self, samples):
        self.__initialize(samples)
        stopping_condition = False
        epoch_counter = 0
        self.__reshuffle_samples(samples)
        while not stopping_condition:
            print("Epoch {}".format(self.epochs))
            for t in range(len(samples)):
                print("Training Iteration {}".format(t))
                bmu = self.__find_bmu(samples[t])
                for row in range(self.Y):
                    for col in range(self.X):
                        self.map[row][col] = self.__update_weight(samples[t], bmu, self.map[row][col])
                self.__update_learning_rate(t)
                self.__update_kernel_width(t)
            epoch_counter += 1
            self.__reshuffle_samples(samples)
            if epoch_counter >= self.epochs:  # lowTrainingError and epochCounter
                stopping_condition = True

    def __initialize(self, samples):
        self.X, self.Y = 4, 4
        self.tau_1 = 8.0
        self.tau_2 = 8.0
        self.kernel_width = 0
        self.initial_kernel_width = 0
        self.learning_rate = 0
        self.initial_learning_rate = 0
        self.map = [[0 for col in range(self.X)] for row in range(self.Y)]
        self.__hypercube_weight_initialization(samples)

    def __find_bmu(self, z):
        distances = list()
        for row in range(self.Y):
            for col in range(self.X):
                w = self.map[row][col]
                distances.append({
                    "dist": euclidean(u=z, v=w),
                    "row": row,
                    "col": col
                })
        lowest_distance = np.Infinity
        for i in range(len(distances)):
            if distances[i]["dist"] < lowest_distance:
                lowest_distance = distances[i]
        return self.map[lowest_distance["row"]][lowest_distance["col"]]

    def __hypercube_weight_initialization(self, samples):
        max_i, max_j = self.__get_max_ij(samples)
        max_k = self.__get_max_k(samples, max_i, max_j)
        max_b = self.__get_max_b(samples, max_i, max_j, max_k)
        if max_i == -1 or max_j == -1 or max_k == -1 or max_b == -1:
            print("Failed to initialize using HyperCube Weight Initialization.")
            print("Defaulting to Uniform Random Value Initialization.")
            self.__random_value_initialization(samples)
            return
        w_y0 = samples[max_i]  # actually w_y1 in literature
        self.map[-1][0] = w_y0
        w_0x = samples[max_j]  # actually w_1x in literature
        self.map[0][-1] = w_0x
        w_00 = samples[max_k]  # actually w_11 in literature
        self.map[0][0] = w_00
        w_yx = samples[max_b]  # actually w_XY in literature
        self.map[-1][-1] = w_yx
        for col in range(1, self.X):  # actually x in range(2, X - 1) in map
            self.map[1][col] = np.add(np.multiply(np.divide(np.subtract(w_0x, w_00), (self.X - 1)), (col - 1)), w_00)
            self.map[-1][col] = np.add(np.multiply(np.divide(np.subtract(w_yx, w_y0), (self.X - 1)), (col - 1)), w_y0)
        for row in range(1, self.Y):  # actually y in range(2, Y - 1) in map
            self.map[row][1] = np.add(np.multiply(np.divide(np.subtract(w_y0, w_00), (self.Y - 1)), (row - 1)), w_00)
            self.map[row][-1] = np.add(np.multiply(np.divide(np.subtract(w_yx, w_0x), (self.Y - 1)), (row - 1)), w_0x)
            for col in range(1, self.X):
                self.map[row][col] = np.add(
                    np.multiply(np.divide(np.subtract(self.map[row][-1], w_y0), (self.X - 1)), (col - 1)), w_y0)

    def __random_value_initialization(self, samples):
        extreme = self.__get_extreme(samples)
        for row in range(self.Y):
            for col in range(self.X):
                self.map[row][col] = self.__get_random_weight_vector(extreme)

    def __update_learning_rate(self, t):
        self.learning_rate = self.initial_learning_rate * np.power(np.e, (-t / self.tau_1))

    def __update_kernel_width(self, t):
        """ Update to the Neighborhood Radius """
        self.kernel_width = self.initial_kernel_width * np.power(np.e, (-t / self.tau_2))

    def __update_weight(self, current_sample, bmu, current_weight):
        return current_weight + self.__change_in_weight(current_sample, current_weight, bmu)

    def __change_in_weight(self, current_sample, current_weight, bmu):
        for i in range(len(current_weight)):
            current_weight[i] = self.__neighborhood_func(bmu, current_weight) * (current_sample[i] - current_weight[i])
        return current_weight

    def __neighborhood_func(self, bmu, current_weight):
        """ Using the Gaussian Kernel approach"""
        dividend = np.power(euclidean(u=bmu, v=current_weight), 2)
        divisor = 2 * np.power(self.kernel_width, 2)
        return self.learning_rate * np.exp(-(dividend / divisor))

    @staticmethod
    def __get_random_weight_vector(extreme):
        col_size = len(extreme)
        weight = [0 for i in range(col_size)]
        for i in range(col_size):
            weight[i] = np.random.uniform(extreme[i]["min"], extreme[i]["max"])
        return weight

    @staticmethod
    def __get_extreme(samples):
        if samples is None:
            print("Cannot compute bounds on empty samples")
            return None
        col_size = len(samples[0])
        extreme = list()
        for c in range(col_size):
            extreme.append({
                "min": np.min(samples[:c]),
                "max": np.max(samples[:c])
            })
        return extreme

    @staticmethod
    def __reshuffle_samples(samples):
        np.random.shuffle(samples)

    @staticmethod
    def __get_max_ij(samples):
        max_dist = -np.Infinity
        max_i, max_j = -1, -1
        sample_size = len(samples)
        for i in range(sample_size):
            for j in range(sample_size):
                if i != j:
                    dist = euclidean(u=samples[i], v=samples[j])
                    if dist > max_dist:
                        max_dist = dist
                        max_i, max_j = i, j
        return max_i, max_j

    @staticmethod
    def __get_max_k(samples, max_i, max_j):
        z_s1 = samples[max_i]
        z_s2 = samples[max_j]
        max_dist, max_k = -np.Infinity, -1
        sample_size = len(samples)
        for k in range(sample_size):
            if k != max_i and k != max_j:
                dist_ik = euclidean(u=z_s1, v=samples[k])
                dist_jk = euclidean(u=z_s2, v=samples[k])
                dist = dist_ik + dist_jk
                if dist > max_dist:
                    max_dist, max_k = dist, k
        return max_k

    @staticmethod
    def __get_max_b(samples, max_i, max_j, max_k):
        z_s1 = samples[max_i]
        z_s2 = samples[max_j]
        z_s3 = samples[max_k]
        max_dist, max_b = -np.Infinity, -1
        sample_size = len(samples)
        for b in range(sample_size):
            if b != max_i and b != max_j and b != max_k:
                dist_ib = euclidean(u=z_s1, v=samples[b])
                dist_jb = euclidean(u=z_s2, v=samples[b])
                dist_kb = euclidean(u=z_s3, v=samples[b])
                dist = dist_ib + dist_jb + dist_kb
                if dist > max_dist:
                    max_dist, max_b = dist, b
        return max_b


def main():
    epochs = 100
    iris = load_iris().data
    train_size = np.floor(len(iris) * 0.75)
    x_train = iris[:train_size]
    # x_test = iris[train_size + 1: -1]
    som = SOM(epochs=epochs)
    som.train(samples=x_train)


if __name__ == '__main__':
    main()
