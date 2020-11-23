import numpy as np
import torch

class EarlyStopMonitor(object):
  def __init__(self, logger, max_round=3, higher_better=True, tolerance=1e-4):
    self.max_round = max_round
    self.num_round = 0
    self.logger = logger
    self.epoch_count = 0
    self.best_epoch = 0

    self.best_ap = 0
    self.best_auc = 0
    self.best_acc = 0
    self.best_loss = 0

    self.best = None
    self.last = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.best is None:
      self.best = curr_val
    elif (curr_val - self.best) / np.abs(self.best) > -self.tolerance:
      self.logger.info(f'Epoch: {self.epoch_count} | Val Metirc continues to decline')
      self.best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1
      self.logger.info(f'Epoch: {self.epoch_count} | Val Metirc stops declining over {self.num_round}/{self.max_round} epoch')

    self.epoch_count += 1
    self.last = curr_val

    return self.num_round >= self.max_round