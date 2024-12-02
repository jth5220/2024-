import numpy as np


class KalmanFilter:
    """
    Kalman Filter Implementation.

    Args:
        transition_matrix (numpy.ndarray): Transition matrix of shape ``(n, n)``.
        measurement_matrix (numpy.ndarray): Measurement matrix of shape ``(m, n)``.
        control_matrix (numpy.ndarray): Control matrix of shape ``(m, n)``.
        process_noise_covariance (numpy.ndarray): Covariance matrix of shape ``(n, n)``.
        measurement_noise_covariance (numpy.ndarray): Covariance matrix of shape ``(m, m)``.
        prediction_covariance (numpy.ndarray): Predicted (a priori) estimate covariance of shape ``(n, n)``.
        initial_state (numpy.ndarray): Initial state of shape ``(n,)``.

    """

    def __init__(
            self,
            transition_matrix,
            measurement_matrix,
            control_matrix=None,
            process_noise_covariance=None,
            measurement_noise_covariance=None,
            prediction_covariance=None,
            initial_state=None
    ):
        self.state_size = transition_matrix.shape[1]
        self.observation_size = measurement_matrix.shape[1]

        self.transition_matrix = transition_matrix
        self.measurement_matrix = measurement_matrix

        self.control_matrix = 0 if control_matrix is None else control_matrix

        self.process_covariance = np.eye(self.state_size) \
            if process_noise_covariance is None else process_noise_covariance

        self.measurement_covariance = np.eye(self.observation_size) \
            if measurement_noise_covariance is None else measurement_noise_covariance

        self.prediction_covariance = np.eye(self.state_size) if prediction_covariance is None else prediction_covariance

        self.x = np.zeros((self.state_size, 1)) if initial_state is None else initial_state

    def predict(self, u=0):
        """
        Prediction step of Kalman Filter.

        Args:
            u (float or int or numpy.ndarray): Control input. Default is `0`.

        Returns:
            numpy.ndarray : State vector of shape `(n,)`.

        """
        self.x = np.dot(self.transition_matrix, self.x) + np.dot(self.control_matrix, u)

        self.prediction_covariance = np.dot(
            np.dot(self.transition_matrix, self.prediction_covariance), self.transition_matrix.T
        ) + self.process_covariance

        return self.x

    def update(self, z):
        """
        Measurement update of Kalman Filter.

        Args:
            z (numpy.ndarray): Measurement vector of the system with shape ``(m,)``.
        """
        y = z - np.dot(self.measurement_matrix, self.x)

        innovation_covariance = np.dot(
            self.measurement_matrix, np.dot(self.prediction_covariance, self.measurement_matrix.T)
        ) + self.measurement_covariance

        optimal_kalman_gain = np.dot(
            np.dot(self.prediction_covariance, self.measurement_matrix.T),
            np.linalg.inv(innovation_covariance)
        )

        self.x = self.x + np.dot(optimal_kalman_gain, y)
        eye = np.eye(self.state_size)
        _t1 = eye - np.dot(optimal_kalman_gain, self.measurement_matrix)
        t1 = np.dot(np.dot(_t1, self.prediction_covariance), _t1.T)
        t2 = np.dot(np.dot(optimal_kalman_gain, self.measurement_covariance), optimal_kalman_gain.T)
        self.prediction_covariance = t1 + t2


def get_process_covariance_matrix(dt):
    """
    Generates a process noise covariance matrix for constant acceleration motion.

    Args:
        dt (float): Timestep.

    Returns:
        numpy.ndarray: Process covariance matrix of shape `(3, 3)`.
    """
    # a = np.array([
    #     [0.25 * dt ** 4, 0.5 * dt ** 3, 0.5 * dt ** 2],
    #     [0.5 * dt ** 3, dt ** 2, dt],
    #     [0.5 * dt ** 2, dt, 1]
    # ])

    a = np.array([
        [dt ** 6 / 36., dt ** 5 / 24., dt ** 4 / 6.],
        [dt ** 5 / 24., 0.25 * dt ** 4, 0.5 * dt ** 3],
        [dt ** 4 / 6., 0.5 * dt ** 3, dt ** 2]
    ])
    return a


def get_transition_matrix(dt):
    """
    Generate the transition matrix for constant acceleration motion.

    Args:
        dt (float): Timestep.

    Returns:
        numpy.ndarray: Transition matrix of shape ``(3, 3)``.

    """
    return np.array([[1., dt, dt * dt * 0.5], [0., 1., dt], [0., 0., 1.]])


class KFTrackerConstantAcceleration(KalmanFilter):
    """
    Kalman Filter with constant acceleration kinematic model.

    Args:
        initial_measurement (numpy.ndarray):  Initial state of the tracker.
        time_step (float) : Time step.
        process_noise_scale (float): Process noise covariance scale.
            or covariance magnitude as scalar value.
        measurement_noise_scale (float): Measurement noise covariance scale.
            or covariance magnitude as scalar value.
    """

    def __init__(self, initial_measurement, time_step=1, process_noise_scale=1.0,
                 measurement_noise_scale=1.0):
        self.time_step = time_step

        measurement_size = initial_measurement.shape[0]
        transition_matrix = np.zeros((3 * measurement_size, 3 * measurement_size))
        measurement_matrix = np.zeros((measurement_size, 3 * measurement_size))
        process_noise_covariance = np.zeros((3 * measurement_size, 3 * measurement_size))
        measurement_noise_covariance = np.eye(measurement_size)
        initial_state = np.zeros((3 * measurement_size,))

        a = get_transition_matrix(self.time_step)
        q = get_process_covariance_matrix(self.time_step)
        for i in range(measurement_size):
            transition_matrix[3 * i:3 * i + 3, 3 * i:3 * i + 3] = a
            measurement_matrix[i, 3 * i] = 1.
            process_noise_covariance[3 * i:3 * i + 3, 3 * i:3 * i + 3] = process_noise_scale * q
            measurement_noise_covariance[i, i] = measurement_noise_scale
            initial_state[i * 3] = initial_measurement[i]

        prediction_noise_covariance = np.ones((3*measurement_size, 3*measurement_size))

        super().__init__(transition_matrix=transition_matrix, measurement_matrix=measurement_matrix,
                         process_noise_covariance=process_noise_covariance,
                         measurement_noise_covariance=measurement_noise_covariance,
                         prediction_covariance=prediction_noise_covariance, initial_state=initial_state)


class KFTracker2D(KFTrackerConstantAcceleration):
    def __init__(self, initial_measurement=np.array([0., 0.]), time_step=1, process_noise_scale=1.0,
                 measurement_noise_scale=1.0):
        assert initial_measurement.shape[0] == 2, initial_measurement.shape
        super().__init__(
            initial_measurement=initial_measurement, time_step=time_step, process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale
        )


class KFTracker4D(KFTrackerConstantAcceleration):
    def __init__(self, initial_measurement=np.array([0., 0., 0., 0.]), time_step=1, process_noise_scale=1.0,
                 measurement_noise_scale=1.0):
        assert initial_measurement.shape[0] == 4, initial_measurement.shape
        super().__init__(
            initial_measurement=initial_measurement, time_step=time_step, process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale
        )


def test_KFTracker2D():
    kf = KFTracker2D(time_step=1)
    print('measurement matrix:')
    print(kf.measurement_matrix)
    print()
    print('process cov:')
    print(kf.process_covariance)
    print()
    print('transition matrix:')
    print(kf.transition_matrix)
    print()
    print('measurement cov:')
    print(kf.measurement_covariance)
    print()
    print('state:')
    print(kf.x)
    print()
    print('predicted measurement:')
    print(np.dot(kf.measurement_matrix, kf.x))
    print()
    print('prediction:')
    print(kf.predict())
    print()
    kf.update(np.array([1.5, 1.5]))
    print('prediction2:')
    print(kf.predict())


if __name__ == '__main__':
    test_KFTracker1D()
    test_KFTracker2D()
