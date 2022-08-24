from comet_ml import Experiment
import numpy as np
from collections import deque
import glob
import pandas as pd
import json
from datetime import datetime

class Logger:
    def __init__(self, comet_ml, tags=None, parameters=None, experiment=None):
        self.stations = 5
        self.comet_ml = comet_ml
        self.logs = pd.DataFrame(columns=["step", "name", "type", "value"])
        self.fname = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

        if self.comet_ml:
            if experiment is None:
                try:
                    json_loc = glob.glob("./**/comet_token.json")[0]
                    with open(json_loc, "r") as f:
                        kwargs = json.load(f)
                except IndexError:
                    kwargs = {
                        "api_key": "XXXXXXXXXXXX",
                        "project_name": "rl-in-wifi",
                        "workspace": "XYZ"
                    }

                self.experiment = Experiment(**kwargs)
            else:
                self.experiment = experiment
        self.sent_mb = 0
        self.speed_window = deque(maxlen=100)
        self.step_time = None
        self.current_speed = 0
        if self.comet_ml:
            if tags is not None:
                self.experiment.add_tags(tags)
            if parameters is not None:
                self.experiment.log_parameters(parameters)

    def log_parameter(self, param_name, param_value):
        if self.comet_ml:
            self.experiment.log_parameter(param_name, param_value)

        entry = {"step": 0, "name": param_name, "type": "parameter", "value": param_value}
        self.logs = self.logs.append(entry, ignore_index=True)

    def log_metric(self, metric_name, value, step=None):
        if self.comet_ml:
            self.experiment.log_metric(metric_name, value, step=step)

        entry = {"step": step, "name": metric_name, "type": "metric", "value": value}
        self.logs = self.logs.append(entry, ignore_index=True)

    def log_metrics(self, metrics, step):
        for metric in metrics:
            self.log_metric(metric, metrics[metric], step=step)

    def begin_logging(self, episode_count, steps_per_ep, sigma, theta, step_time):
        self.step_time = step_time
        self.log_parameter("Episode count", episode_count)
        self.log_parameter("Steps per episode", steps_per_ep)
        self.log_parameter("theta", theta)
        self.log_parameter("sigma", sigma)
        self.log_parameter("Step time", step_time)

    def log_round(self, states, reward, cumulative_reward, info, loss, observations, step):
        if self.comet_ml:
            self.experiment.log_histogram_3d(states, name="Observations", step=step)
        info = [[j for j in i.split("|")] for i in info]
        info = np.mean(np.array(info, dtype=np.float32), axis=0)
        try:
            round_mb = info[0]
        except Exception as e:
            print(info)
            print(reward)
            raise e
        self.speed_window.append(round_mb)
        self.current_speed = np.mean(np.asarray(self.speed_window)/self.step_time)
        self.sent_mb += round_mb
        CW = info[1]
        self.stations = info[2]
        fairness = info[3]

        self.log_metric("Round reward", np.mean(reward), step=step)
        self.log_metric("Per-ep reward", np.mean(cumulative_reward), step=step)
        self.log_metric("Megabytes sent", self.sent_mb, step=step)
        self.log_metric("Round megabytes sent", round_mb, step=step)
        self.log_metric("Chosen CW", CW, step=step)
        self.log_metric("Station count", self.stations, step=step)
        self.log_metric("Current throughput", self.current_speed, step=step)
        self.log_metric("Fairness index", fairness, step=step)

        for i, obs in enumerate(observations):
            self.log_metric(f"Observation {i}", obs, step=step)
            self.log_metrics(loss, step=step)
        
    def log_episode(self, cumulative_reward, speed, step):
        self.log_metric("Cumulative reward", cumulative_reward, step=step)
        self.log_metric("Speed", speed, step=step)

        self.sent_mb = 0
        self.last_speed = speed
        self.speed_window = deque(maxlen=100)
        self.current_speed = 0
        self.logs.to_csv(self.fname)

    def end(self):
        if self.comet_ml:
            self.experiment.end()
        
        self.logs.to_csv(self.fname)