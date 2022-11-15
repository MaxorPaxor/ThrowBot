import numpy as np
import time
import os
import pickle
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env.robot_env_dt import RoboticArm


class BallisticModule:
    def __init__(self, arm):
        self.robotic_arm = arm
        self.residual_net = Residual(input_size=1, hidden_size=32, output_size=1, num_hidden_layers=0, lr=3e-04)

        # Buffer
        self.MAX_MEMORY = 10_000
        self.memory = deque(maxlen=int(self.MAX_MEMORY))  # popleft()

    def ballistic_throw(self, distance):
        """
        Analytical Ballistic Model Throw
        Input: Distance [0.8 - 2.3]
        """

        # Theoretical velocity
        velocity = self.ballistic_model(distance)

        # Numerical Mapping between v and dt
        dt = self.calib_distance(velocity)

        # Throw
        traj1 = self.robotic_arm.trajectory(j1=0.0, j2=0.5, j3=-0.3, j4=0.0,
                                            j5=-1.5, j6=0.0, gripper=1.0, dt=dt)
        self.robotic_arm.pub_command.publish(traj1)
        # self.rate.sleep()
        time.sleep(dt)

        traj2 = self.robotic_arm.trajectory(j1=0.0, j2=0.7, j3=0.1, j4=0.0,
                                            j5=-1.2, j6=0.0, gripper=1.0, dt=dt)
        self.robotic_arm.pub_command.publish(traj2)
        # self.rate.sleep()
        time.sleep(dt)

        traj3 = self.robotic_arm.trajectory(j1=0.0, j2=0.8, j3=0.4, j4=0.0,
                                            j5=-0.9, j6=0.0, gripper=1.0, dt=dt)
        self.robotic_arm.pub_command.publish(traj3)
        # self.rate.sleep()
        time.sleep(dt)

        traj4 = self.robotic_arm.trajectory(j1=0.0, j2=0.9, j3=0.7, j4=0.0,
                                            j5=-0.586, j6=0.0, gripper=-1.0, dt=dt)
        self.robotic_arm.pub_command.publish(traj4)
        # self.rate.sleep()
        time.sleep(dt)

        self.robotic_arm.wait_for_object_to_touch_ground()
        land_pos = self.robotic_arm.object_position[0]

        # print(f"Target: {distance}")
        # print(f"Landing Position: {land_pos}")
        # print(f"Theoretical Velocity: {velocity}")
        # print(f"dt: {dt}")
        # print(f"---")

        return land_pos, velocity

    @staticmethod
    def ballistic_model(distance):
        """
        Analytical Ballistic Model Throw
        Input: Distance [0.9 - 2.0]
        Output: Velocity v
        """
        release_x = 0.813
        release_z = 0.349
        position_x = distance
        g = 9.81

        v_nom = g * (position_x - release_x) ** 2
        v_denom = position_x - release_x + release_z
        v = np.sqrt(v_nom / v_denom)
        return v

    @staticmethod
    def calib_distance(distance):
        # Calculate bias
        # x = np.array([5.04, 4.55, 3.63, 3.22, 2.86, 2.64, 2.33, 2.19, 1.94, 1.74, 1.66,
        #               1.52, 1.41, 1.34, 1.29, 1.2, 1.16, 1.14, 1.01, 0.76, 0.58, 0.24])
        # y = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
        #               0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.25, 0.3, 0.4, 0.5])
        # z = np.polyfit(x, y, deg=6)
        # p = np.poly1d(z)
        # print(z)

        z = np.array(
            [-0.00135605,  0.02118723, -0.12275126,  0.30929411, -0.23891526, -0.3051288, 0.587706])
        p = np.poly1d(z)
        dt = p(distance)

        return dt

    def evaluate_ballistic(self):

        dx = 0.05
        target_list = np.arange(0.8, 2.3 + dx, dx)
        error_list = []

        for target in target_list:
            self.robotic_arm.reset()
            landing, _ = self.ballistic_throw(distance=target)
            error = abs(landing - target)
            error_list.append(error)

        mean_error = np.mean(np.array(error_list))
        print(f"Mean Error: {mean_error}")
        return mean_error

    def collect_data(self, number_of_data=500):

        print(f"Collecting Data...")
        for throw in range(number_of_data):
            self.robotic_arm.reset()
            target = np.random.rand() * 1.5 + 0.8
            landing, vel = self.ballistic_throw(distance=target)
            vel_landing_bm = self.ballistic_model(landing)
            res_label = vel - vel_landing_bm
            self.memory.append([landing, res_label])

            print(f"{throw+1} / {number_of_data}")
            print(f"Pos: {landing}, Res: {res_label}, ")
            print(f"BM(landing): {vel_landing_bm}, Res: {res_label}, vel: {vel}")
            print(f"---")

        pickle.dump(self.memory, open('data/memory_res.pkl', 'wb'))


class Residual(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2, lr=3e-04):
        super(Residual, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.linear_input = nn.Linear(input_size, hidden_size)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)])
        self.linear_output = nn.Linear(hidden_size, output_size)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-2)  # weight_decay=1e-2
        self.criterion = nn.MSELoss()
        # warmup_steps = 5_000
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lambda steps: min((steps + 1) / warmup_steps, 1)
        # )

    def forward(self, distance):
        x = distance

        x = self.linear_input(x)
        x = F.relu(x)

        for i, l in enumerate(self.linear_hidden):
            x = l(x)
            x = F.relu(x)

        x = self.linear_output(x)

        return x

    def save(self, file_name='residual.pth'):
        model_folder_path = './weights'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)

        torch.save({'state_dict': self.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, file_name)

    def load(self):
        checkpoint = torch.load("./weights/residual.pth", map_location=torch.device('cpu'))

        try:
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        except KeyError:
            self.load_state_dict(checkpoint)


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        super(MyDataSet, self).__init__()
        # store the raw tensors
        self._x = data[:, 0]
        self._y = data[:, 1]

        self.transform = transform

    def __len__(self):
        # a DataSet must know it size
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]

        return x, y


def train_res():
    residual_net = Residual(input_size=1, hidden_size=32, output_size=1, num_hidden_layers=0, lr=3e-04)
    data = list(pickle.load(open('data/memory_res.pkl', 'rb')))
    data = torch.tensor(data)
    ds = MyDataSet(data)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            residual_net.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = residual_net(inputs)
            loss = residual_net.criterion(outputs, labels)
            loss.backward()
            residual_net.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    residual_net.save()


if __name__ == '__main__':
    robotic_arm = RoboticArm()
    ballistic = BallisticModule(arm=robotic_arm)
    # ballistic.evaluate_ballistic()
    ballistic.collect_data()

    # train_res()
