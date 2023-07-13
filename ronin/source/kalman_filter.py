import numpy as np
from scipy.spatial.transform import Rotation


class ExtendedKalmanFilter:
    """Extended Kalman Filter for tracking a human with IMU measurements"""

    def __init__(self, x, P):
        """Initialize the Kalman Filter

        Args:
            x (numpy.array): Initial state estimate
            P (numpy.array): Initial error covariance matrix
        """
        self.x = x  # State estimate
        self.P = P  # Error covariance matrix

    def update(self, z, R):
        """Update the state estimate based on sensor measurements

        Args:
            z (numpy.array): Measurement vector - [position_x, position_y, roll, pitch, yaw]
            R (numpy.array): Measurement noise covariance matrix {5x5}
        """
        # Compute Kalman gain
        H = np.eye(5,7)  # Jacobian of observation function

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        
        # Update state estimate
        z_ = self.x[:5]
        self.x = self.x + K @ (z - z_)

        # Update error covariance matrix
        self.P = self.P - K @ H @ self.P

    def propagate(self, u, dt, Q):
        """Propagate the state estimate based on the motion model

        Args:
            u (numpy.array): Control input vector - [acceleration_x, acceleration_y, acceleration_z, roll_rate, pitch_rate, yaw_rate]
            dt (float): Time interval in seconds
            Q (numpy.array): Process noise covariance matrix {6x6} (based on control inputs)
        """
        # Propagate state estimate
        position_x, position_y, roll, pitch, yaw = self.x[:5]

        # Update roll, pitch, and yaw angles
        old_rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        updated_rotation = old_rotation * Rotation.from_euler('xyz', u[3:]*dt, degrees=False)
        roll, pitch, yaw = updated_rotation.as_euler('xyz')

        # Rotate acceleration to the global frame
        acceleration_body = u[:3]
        acceleration_global = updated_rotation.apply(acceleration_body)
        acceleration_x, acceleration_y, acceleration_z = acceleration_global

        # Update velocity
        velocity_x, velocity_y = self.x[5:7]
        velocity_x += acceleration_x * dt
        velocity_y += acceleration_y * dt

        # Update position
        displacement_x = velocity_x * dt
        displacement_y = velocity_y * dt

        position_x += displacement_x
        position_y += displacement_y

        # Update state estimate
        self.x[:5] = np.array([position_x, position_y, roll, pitch, yaw])
        self.x[5:7] = np.array([velocity_x, velocity_y])


        G = np.eye(7, 6)  # Jacobian of state transition function, anyhow initialised...

        G[:2, 4:6] = dt * np.eye(2)
        G[2:5, 2:5] = updated_rotation.as_matrix()
        G[5:7, 2:4] = np.array([[0, -u[5] * dt], [u[5] * dt, 0]])



        self.P = G @ Q @ G.T + self.P
