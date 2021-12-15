import numpy as np
import matplotlib.pyplot as plt


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.45, theta=0.1, dt=2e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def test_noise(self, interval=100):
        x_list = []
        normal_noise_list = []
        mu = 0
        x_prev = 0
        for i in range(interval):
            normal_noise = np.random.normal(size=1)[0]
            x = x_prev + self.theta * (mu - x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * normal_noise
            x_prev = x
            x_list.append(x)
            normal_noise_list.append(normal_noise)

        # Plot Noise behavior
        plt.plot(range(interval), x_list)
        plt.plot(range(interval), normal_noise_list)
        plt.legend(["Ornstein Noise", "Normal Noise"])
        plt.show(block=True)


if __name__ == "__main__":
    # Test Noise:
    # noise = OUActionNoise(mu=0, sigma=0.8, dt=3e-2, theta=0.1, x0=None)
    noise = OUActionNoise(mu=0, sigma=1.0, dt=4e-2, theta=0.0, x0=None)
    noise.test_noise(interval=10)
