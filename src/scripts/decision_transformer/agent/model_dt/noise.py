import numpy as np
import matplotlib.pyplot as plt


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.45, theta=0.1, dt=2e-2, x0=None):
        # dt = 1e-1, theta = 0.02, sigma = 0.2
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
        normal_noise_list = []
        noise_list = []
        noise_list_no_theta = []
        noise_list_2xdt = []

        mu = 0
        x_prev = 0
        x_prev_no_theta = 0
        x_prev_2xdt = 0
        for i in range(interval):
            normal_noise = np.random.normal(size=1)[0]

            x = x_prev + self.theta * (mu - x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * normal_noise
            x_no_theta = x_prev_no_theta + \
                self.sigma * np.sqrt(self.dt) * normal_noise
            x_2xdt = x_prev_2xdt + self.theta * (mu - x_prev) * self.dt * 2 + \
                self.sigma * np.sqrt(2*self.dt) * normal_noise
            x_prev = x
            x_prev_no_theta = x_no_theta
            x_prev_2xdt = x_2xdt

            noise_list.append(x)
            normal_noise_list.append(normal_noise)
            noise_list_no_theta.append(x_prev_no_theta)
            noise_list_2xdt.append(x_2xdt)

        # Plot Noise behavior
        plt.plot(range(interval), normal_noise_list)
        plt.plot(range(interval), noise_list)
        plt.plot(range(interval), noise_list_no_theta)
        plt.plot(range(interval), noise_list_2xdt)
        plt.legend(["Normal Noise", "Ornstein Noise", "Ornstein Noise No Theta", "2x dt"])
        plt.show(block=True)


if __name__ == "__main__":
    # Test Noise:
    # noise = OUActionNoise(mu=0, sigma=0.8, dt=3e-2, theta=0.1, x0=None)
    noise = OUActionNoise(mu=0, sigma=1.0, dt=2e-1, theta=0, x0=None)
    noise.test_noise(interval=10)
