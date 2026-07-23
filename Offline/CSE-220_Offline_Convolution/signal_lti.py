import numpy as np


def readable_time_ticks(time_values, max_labels=18):
    if len(time_values) <= max_labels:
        return time_values

    step = int(np.ceil(len(time_values) / max_labels))
    ticks = time_values[::step]

    if ticks[-1] != time_values[-1]:
        ticks.append(time_values[-1])

    return ticks


class DiscreteSignal:
    """Finite discrete-time signal with integer indices."""

    # Create a finite discrete-time signal over the given integer range.
    def __init__(self, start_time, end_time):
        # raise NotImplementedError("Complete the DiscreteSignal constructor")
        self.start_time = int(start_time)
        self.end_time = int(end_time)
        self.values = np.zeros(end_time - start_time + 1, dtype=float)

    # Return the number of stored samples in the signal.
    def __len__(self):
        # raise NotImplementedError("Complete __len__")
        return len(self.values)

    # Return the integer time indices covered by the signal.
    def times(self):
        # raise NotImplementedError("Complete times")
        return range(self.start_time, self.end_time + 1) 

    # Return the signal value at the given time index.
    def get_value_at_time(self, t):
        # raise NotImplementedError("Complete get_value_at_time")
        if self.start_time <= t <= self.end_time:
            return self.values[t - self.start_time]
        return 0.0

    # Set the signal value at the given time index.
    def set_value_at_time(self, t, value):
        # raise NotImplementedError("Complete set_value_at_time")
        if self.start_time <= t <= self.end_time:
            self.values[t - self.start_time] = value

    # Return a shifted copy of the signal.
    def shift(self, k):
        # raise NotImplementedError("Complete shift")
        shifted_signal = DiscreteSignal(self.start_time + k, self.end_time + k)
        shifted_signal.values = np.copy(self.values)
        return shifted_signal

    # Return the sum of this signal and another signal.
    def add(self, other):
        # raise NotImplementedError("Complete add")
        new_start = min(self.start_time, other.start_time)
        new_end = max(self.end_time, other.end_time)
        sum_signal = DiscreteSignal(new_start, new_end)
        for n in sum_signal.times():
            sum_signal.set_value_at_time(n, self.get_value_at_time(n) + other.get_value_at_time(n))
        return sum_signal            

    # Return a scaled copy of the signal.
    def multiply(self, scalar):
        # raise NotImplementedError("Complete multiply")
        scaled_signal = DiscreteSignal(self.start_time, self.end_time)
        scaled_signal.values = float(scalar) * self.values
        return scaled_signal

    # Return the nonzero samples of the signal.
    def nonzero_samples(self, tolerance=1e-12):
        # raise NotImplementedError("Complete nonzero_samples")
        samples = []
        for n in self.times():
            value = self.get_value_at_time(n)
            if abs(value) > tolerance:
                samples.append((n, value))
        return samples 

    def plot(self, title, save_path=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        time_values = list(self.times())
        markerline, stemlines, baseline = ax.stem(time_values, self.values)
        markerline.set_markersize(6)
        baseline.set_color("black")
        baseline.set_linewidth(1)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.35)
        ax.set_xticks(readable_time_ticks(time_values))
        ax.tick_params(axis="x", labelsize=9)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return ax


class LTISystem:
    """Discrete-time LTI system described by a finite impulse response."""

    # Store the impulse response that defines the LTI system.
    def __init__(self, impulse_response):
        # raise NotImplementedError("Complete the LTISystem constructor")
        self.impulse_response = impulse_response

    # Return the output time range for the convolution result.
    def output_range(self, input_signal):
        # raise NotImplementedError("Complete output_range")
        start_time = input_signal.start_time + self.impulse_response.start_time
        end_time = input_signal.end_time + self.impulse_response.end_time
        return start_time, end_time
    
    # Return all shifted and scaled impulse-response components for the input.
    
    # For every nonzero input sample x[k], the system's response to the single
    # scaled impulse x[k]*delta[n-k] is x[k]*h[n-k]: h shifted by k, scaled by x[k].
    def get_response_components(self, input_signal):
        # raise NotImplementedError("Complete get_response_components")
        h = self.impulse_response
        components = []
        for k, x_k in input_signal.nonzero_samples():
            component = h.shift(k).multiply(x_k)
            components.append(component)
        return components

    # Return the system output using superposition of response components.
    def output_by_superposition(self, input_signal):
        # raise NotImplementedError("Complete output_by_superposition")
        start, end = self.output_range(input_signal)
        result = DiscreteSignal(start, end)
        for component in self.get_response_components(input_signal):
            result = result.add(component)
        return result

    # Return the nonzero product terms that contribute to one output sample.
    def get_contributions_at_time(self, input_signal, n):
        # raise NotImplementedError("Complete get_contributions_at_time")
        h = self.impulse_response
        contributions = []
        for k, x_k in input_signal.nonzero_samples():
            h_value = h.get_value_at_time(n - k)
            if h_value != 0.0:
                contributions.append((k, x_k, h_value, x_k * h_value))
        return contributions

    # Return one output sample of the LTI system.
    def output_at_time(self, input_signal, n):
        # raise NotImplementedError("Complete output_at_time")
        return sum(
            term for _, _, _, term in self.get_contributions_at_time(input_signal, n)
        )

    # Return the complete output signal of the LTI system.
    def output(self, input_signal):
        # raise NotImplementedError("Complete output")
        start, end = self.output_range(input_signal)
        result = DiscreteSignal(start, end)
        for n in range(start, end + 1):
            result.set_value_at_time(n, self.output_at_time(input_signal, n))
        return result
