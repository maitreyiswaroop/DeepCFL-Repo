from abc import ABC, abstractmethod
import math
from matplotlib import pyplot as plt


class HyperParamScheduler(ABC):
    """
    Base class for hyper-parameter schedulers.
    """

    def __init__(self, init_val: float) -> None:
        super().__init__()

        self.val = init_val
        self.init_val = init_val
        self.t = 0

    def step(self) -> float:
        """
        Return the value of the scheduler at the next time step.
        """
        self.t += 1
        self.val = self.update(self.val, self.t)
        return self.val

    def visualize(self, num_steps: int, save=None) -> None:
        """
        Return a list of values of the scheduler for the next `num_steps` time steps.
        """
        # reset figure shape and size
        plt.figure(figsize=(6, 4))
        vals = [self.init_val] + [self.step() for _ in range(num_steps)]
        plt.plot(range(num_steps + 1), vals)
        # plt.show()
        if save is not None:
            plt.savefig(save)
            plt.clf()


    @abstractmethod
    def update(self, current_val: float, next_t: int) -> float:
        """
        Update the value of the scheduler based on it's current value and the next time step.
        """
        pass


class ConstantScheduler(HyperParamScheduler):
    def __init__(self, init_val: float) -> None:
        super().__init__(init_val)

    def update(self, current_val: float, next_t: int) -> float:
        return current_val


class LinearScheduler(HyperParamScheduler):
    def __init__(self, init_val: float, final_val: float, num_steps: int) -> None:
        super().__init__(init_val)

        self.final_val = final_val
        self.num_steps = num_steps

    def update(self, current_val: float, next_t: int) -> float:
        alpha = min(next_t / self.num_steps, 1)
        return (1 - alpha) * self.init_val + alpha * self.final_val


class ExponentialScheduler(HyperParamScheduler):
    def __init__(self, init_val: float, final_val: float, half_life: float) -> None:
        super().__init__(init_val)

        self.final_val = final_val
        self.half_life = half_life
        self.decay_rate = math.log(2) / half_life
        self.y0 = init_val - final_val

    def update(self, current_val: float, next_t: int) -> float:
        return self.y0 * math.exp(-self.decay_rate * next_t) + self.final_val


class SigmoidScheduler(HyperParamScheduler):
    def __init__(
        self, init_val: float, final_val: float, width: float, start: float
    ) -> None:
        super().__init__(init_val)

        self.final_val = final_val
        self.width = width
        self.start = start

        self.x_scale = width / 10  # A normal sigmoid has a width of about 10
        self.x_offset = width / 2 - start * width
        self.y_scale = final_val - init_val

        self.init_val = self.update(init_val, 0)

    def update(self, current_val: float, next_t: int) -> float:
        x = (next_t - self.x_offset) / self.x_scale
        return sigmoid(x) * self.y_scale + self.init_val


class CosineScheduler(HyperParamScheduler):
    def __init__(self, init_val: float, final_val: float, period: float) -> None:
        super().__init__(init_val)

        self.final_val = final_val
        self.period = period

    def update(self, current_val: float, next_t: int) -> float:
        return (self.init_val - self.final_val) * (
            1 + math.cos(2 * math.pi * next_t / self.period)
        ) / 2 + self.final_val


class ExponentialSinusoidScheduler(HyperParamScheduler):
    def __init__(
        self,
        init_val: float,
        final_val: float,
        half_life: float,
        sin_amplitude: float,
        sin_period: float,
        sin_half_life: float,
    ) -> None:
        super().__init__(init_val)

        self.final_val = final_val
        self.half_life = half_life
        self.decay_rate = math.log(2) / half_life
        self.y0 = init_val - final_val
        self.sin_amplitude = sin_amplitude
        self.sin_period = sin_period
        self.sin_half_life = sin_half_life
        self.sin_decay_rate = math.log(2) / sin_half_life

    def update(self, current_val: float, next_t: int) -> float:
        y_decay = self.y0 * math.exp(-self.decay_rate * next_t) + self.final_val
        sin_amplitude = self.sin_amplitude * math.exp(-self.sin_decay_rate * next_t)
        y_sin = sin_amplitude * math.sin(2 * math.pi * next_t / self.sin_period)
        return y_decay + y_sin


class SigmoidSinusoidScheduler(HyperParamScheduler):
    def __init__(
        self,
        init_val: float,
        final_val: float,
        width: float,
        start: float,
        sin_amplitude: float,
        sin_period: float,
        sin_half_life: float,
    ) -> None:
        super().__init__(init_val)

        self.final_val = final_val
        self.width = width
        self.start = start
        self.sin_amplitude = sin_amplitude
        self.sin_period = sin_period
        self.sin_half_life = sin_half_life
        self.sin_decay_rate = math.log(2) / sin_half_life

        self.x_scale = width / 10  # A normal sigmoid has a width of about 10
        self.x_offset = width / 2 - start * width
        self.y_scale = final_val - init_val

        self.init_val = self.update(init_val, 0)

    def update(self, current_val: float, next_t: int) -> float:
        x_sig = (next_t - self.x_offset) / self.x_scale
        y_sig = sigmoid(x_sig) * self.y_scale + self.init_val
        x_sin_amplitude = math.fabs(next_t - self.x_offset)
        sin_amplitude = self.sin_amplitude * math.exp(
            -self.sin_decay_rate * x_sin_amplitude
        )
        y_sin = sin_amplitude * math.sin(2 * math.pi * next_t / self.sin_period)
        return y_sig + y_sin

class SigmoidSinusoidSchedulerLag(HyperParamScheduler):
    def __init__(
        self,
        init_val: float,
        final_val: float,
        width: float,
        start: float,
        sin_amplitude: float,
        sin_period: float,
        sin_half_life: float,
        lag: float,
    ) -> None:
        super().__init__(init_val)

        self.final_val = final_val
        self.width = width
        self.start = start
        self.sin_amplitude = sin_amplitude
        self.sin_period = sin_period
        self.sin_half_life = sin_half_life
        self.sin_decay_rate = math.log(2) / sin_half_life
        self.lag=lag

        self.x_scale = width / 10  # A normal sigmoid has a width of about 10
        self.x_offset = width / 2 - start * width
        self.y_scale = final_val - init_val

        self.init_val = self.update(init_val, 0)

    def update(self, current_val: float, next_t: int) -> float:
        x_sig = (next_t - self.x_offset) / self.x_scale
        y_sig = sigmoid(x_sig) * self.y_scale + self.init_val
        x_sin_amplitude = math.fabs(next_t - self.x_offset)
        sin_amplitude = self.sin_amplitude * math.exp(
            -self.sin_decay_rate * x_sin_amplitude
        )
        y_sin = sin_amplitude * math.sin(2 * math.pi * next_t / self.sin_period)
        if self.lag>0:
            self.lag-=1
            return y_sig
        else:
            return abs(y_sig + y_sin)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))