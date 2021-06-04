import copy
import collections
import time
import ctypes
import multiprocessing as mp
import dill
import logging
import os
from logging import getLogger

import torch
import torch.nn.functional as F
import numpy as np

import pfrl
from pfrl import agent
from pfrl.utils.batch_states import batch_states
from pfrl.utils.contexts import evaluating
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.replay_buffer import batch_experiences
from pfrl.replay_buffer import batch_recurrent_experiences
from pfrl.replay_buffer import ReplayUpdater
from pfrl.utils.recurrent import get_recurrent_state_at
from pfrl.utils.recurrent import mask_recurrent_state_at
from pfrl.utils.recurrent import one_step_forward
from pfrl.utils.recurrent import pack_and_forward
from pfrl.utils.recurrent import recurrent_state_as_numpy
from pfrl.replay_buffer import AbstractEpisodicReplayBuffer

from pfrl.wrappers.atari_wrappers import LazyFrames

import backtracking
from .dqn import DQN

def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def compute_value_loss(y, t, clip_delta=True, batch_accumulator="mean"):
    """Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        return F.smooth_l1_loss(y, t, reduction=batch_accumulator)
    else:
        return F.mse_loss(y, t, reduction=batch_accumulator) / 2


def compute_weighted_value_loss(
    y, t, weights, clip_delta=True, batch_accumulator="mean"
):
    """Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        weights (torch.Tensor): Weights for y, t.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        losses = F.smooth_l1_loss(y, t, reduction="none")
    else:
        losses = F.mse_loss(y, t, reduction="none") / 2
    losses = losses.reshape(-1,)
    weights = weights.to(losses.device)
    loss_sum = torch.sum(losses * weights)
    if batch_accumulator == "mean":
        loss = loss_sum / y.shape[0]
    elif batch_accumulator == "sum":
        loss = loss_sum
    return loss


def _batch_reset_recurrent_states_when_episodes_end(
    batch_done, batch_reset, recurrent_states
):
    """Reset recurrent states when episodes end.

    Args:
        batch_done (array-like of bool): True iff episodes are terminal.
        batch_reset (array-like of bool): True iff episodes will be reset.
        recurrent_states (object): Recurrent state.

    Returns:
        object: New recurrent states.
    """
    indices_that_ended = [
        i
        for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
        if done or reset
    ]
    if indices_that_ended:
        return mask_recurrent_state_at(recurrent_states, indices_that_ended)
    else:
        return recurrent_states

class EBUDQN(DQN):
    """Deep Q-Network algorithm.

    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.Recurrent` and is updated in a recurrent
            manner.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
    """

    saved_attributes = ("model", "target_model", "optimizer")

    def __init__(
        self,
        q_function,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=None,
        replay_start_size=50000,
        minibatch_size=32,
        update_interval=1,
        target_update_interval=10000,
        clip_delta=True,
        phi=lambda x: x,
        target_update_method="hard",
        soft_update_tau=1e-2,
        n_times_update=1,
        batch_accumulator="mean",
        episodic_update_len=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        recurrent=False,
        max_grad_norm=None,
        replay_buffer_snapshot_path=None,
        replay_buffer_snapshot_interval=100000,
        **kwargs,
    ):
        self.model = q_function
        # TODO: make it more general
        self.n_actions = self.model[-2].out_features

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
       
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.clip_delta = clip_delta
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ("mean", "sum")
        self.logger = logger
        self.batch_states = batch_states
        self.recurrent = recurrent
        if self.recurrent:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
       
        self.minibatch_size = minibatch_size
        self.episodic_update_len = episodic_update_len
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval
        self.n_times_update = n_times_update
        self.max_grad_norm = max_grad_norm
        self.replay_buffer_snapshot_path = replay_buffer_snapshot_path
        self.replay_buffer_snapshot_interval = replay_buffer_snapshot_interval

        assert (
            target_update_interval % update_interval == 0
        ), "target_update_interval should be a multiple of update_interval"

        self.t = 0
        self.optim_t = 0  # Compensate pytorch optim not having `t`
        self._cumulative_steps = 0
        self.last_state = None
        self.last_action = None
        self.target_model = None
        self.sync_target_network()

        # Statistics
        self.q_record = collections.deque(maxlen=1000)
        self.loss_record = collections.deque(maxlen=100)

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

        # Error checking
        if (
            self.replay_buffer.capacity is not None
            and self.replay_buffer.capacity < self.replay_start_size
        ):
            raise ValueError("Replay start size cannot exceed replay buffer capacity.")

        # EBU related
        self.batchnum = 0
        self.batch_count = 0
        self.beta = 0.5
        self.discount = self.gamma

    def update(self, exp_batch, errors_out=None):
        """Update the model from experiences

        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.

        Returns:
            None
        """
        has_weight = "weights" in exp_batch
        if has_weight:            
            exp_batch["weights"] = torch.tensor(
                [elem[0]["weight"] for elem in experiences],
                device=self.device,
                dtype=torch.float32,
            )
            if errors_out is None:
                errors_out = []

        loss = self._compute_loss(exp_batch, errors_out=errors_out)
        if has_weight:
            if hasattr(self.replay_buffer, 'update_errors'):
                self.replay_buffer.update_errors(errors_out)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1

    def _compute_y(self, exp_batch):
        batch_size = exp_batch["state"].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch["state"]

        if self.recurrent:
            qout, _ = pack_and_forward(
                self.model, batch_state, exp_batch["recurrent_state"]
            )
        else:
            qout = self.model(batch_state)

        batch_actions = exp_batch["action"]
        batch_q = torch.reshape(qout.evaluate_actions(batch_actions), (batch_size, 1))

        return batch_q

    def _compute_t(self, states):
        return self.target_model(states).q_values

    def _compute_loss(self, exp_batch, errors_out=None):
        """Compute the Q-learning loss for a batch of experiences


        Args:
          exp_batch (dict): A dict of batched arrays of transitions
        Returns:
          Computed loss from the minibatch of experiences
        """
        y = self._compute_y(exp_batch)
        t = exp_batch['target']
        
        self.q_record.extend(y.detach().cpu().numpy().ravel())

        if errors_out is not None:
            #del errors_out[:]
            delta = torch.abs(y - t)

            if delta.ndim == 2:
                delta = torch.sum(delta, dim=1)
                
            delta = delta.detach().cpu().numpy()
            for i, e in enumerate(delta):
                errors_out.append(e)

        if "weights" in exp_batch:
            return compute_weighted_value_loss(
                y,
                t,
                exp_batch["weights"],
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator,
            )        
        else:
            return compute_value_loss(
                y,
                t,
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator,
            )

    def random_episode(self):        
        done_array = np.asarray([e[0]['is_state_terminal'] for e in self.replay_buffer.memory])
        terminal_array = np.where(done_array==True)[0]       

        batchnum = 0
        while batchnum == 0:
            # exclude some early and final episodes from sampling due to indexing issues,
            # sample two episodes (ind1 for main, and ind2 for the remaining steps to make multiple of 32)
            ind = np.random.choice(range(len(terminal_array)), 2)
            ind1 = ind[0]
            ind2 = ind[1]

            indice_array = range(terminal_array[ind1],terminal_array[ind1-1]+3,-1)		
            epi_len = len(indice_array)		
            batchnum = int(np.ceil(epi_len/float(self.minibatch_size)))

        remainindex = int(batchnum * self.minibatch_size - epi_len)

        # Normally an episode does not have steps=multiple of 32.
        # Fill last minibatch with redundant steps from another episode
        indice_array= np.append(indice_array, range(terminal_array[ind2], terminal_array[ind2]-remainindex, -1))
        indice_array = indice_array.astype(int)

        epi_len = len(indice_array)
        rewards = np.asarray([self.replay_buffer.memory[i][0]['reward'] for i in indice_array])
        actions = np.asarray([self.replay_buffer.memory[i][0]['action'] for i in indice_array])
        terminals = np.asarray([self.replay_buffer.memory[i][0]['is_state_terminal'] for i in indice_array])
        if isinstance(self.replay_buffer.memory[0][0]['state'], LazyFrames):
            state1 = np.asarray([self.replay_buffer.memory[i][0]['state'].__array__() for i in indice_array])
        else:
            state1 = np.asarray([self.replay_buffer.memory[i][0]['state'] for i in indice_array])

        return state1, actions, rewards, batchnum, terminals

    def _do_training(self):
        # if all minibatches of previous episode is updated,
        # sample a new episode to create a new temporary target Q-table, Q_tilde
        if self.batchnum == self.batch_count:
            self.Q_tilde = np.array([]) # temporary target Q-table of next states S'
            self.epi_state, self.epi_actions, self.epi_rewards, self.batchnum, self.epi_terminals = self.random_episode() #sample a new episode
            self.epi_len = self.batchnum * self.minibatch_size
            
            with torch.no_grad():
                for i in range(self.batchnum):
                    epi_state_pth = batch_states(self.epi_state[self.minibatch_size * i:self.minibatch_size * (i+1)], 
                                            device=self.device,
                                            phi=self.phi)
                    self.Q_tilde = np.append(self.Q_tilde, self._compute_t(epi_state_pth).cpu().numpy())
            self.Q_tilde = np.reshape(self.Q_tilde, (self.batchnum * self.minibatch_size, self.n_actions))
            self.Q_tilde = np.roll(self.Q_tilde, self.n_actions) # Q(S2) becomes the first column
            for i in range(self.epi_len):
                if self.epi_terminals[i]:
                    self.Q_tilde[i,:] = [0]*self.n_actions

            self.y_ = np.zeros(self.epi_len,dtype=np.float32) #target value

            for i in range(0, self.epi_len):
                if i < self.epi_len - 1:
                    # The last minibatch stores some redundant transitions of the second episode to fill a minibatch,
                    # so a terminal most likely occurs before self.epi_len
                    if self.epi_terminals[i]:
                        self.y_[i] = self.epi_rewards[i]
                        self.Q_tilde[i+1,self.epi_actions[i]] = self.beta * self.y_ [i] + (1-self.beta)*self.Q_tilde[i+1,self.epi_actions[i]]
                    elif self.epi_terminals[i+1]:
                        self.y_[i] = self.epi_rewards[i] + self.discount * np.max(self.Q_tilde[i])
                        self.Q_tilde[i+1,:] = 0
                    else:
                        self.y_[i] = self.epi_rewards[i] + self.discount * np.max(self.Q_tilde[i])
                        self.Q_tilde[i+1, self.epi_actions[i]] = self.beta * self.y_ [i] + (1-self.beta) * self.Q_tilde[i+1,self.epi_actions[i]]
                if i == self.epi_len - 1: #Most likely to be a transition of a redundant episode
                    if self.epi_terminals[i]:
                        self.y_[i] = self.epi_rewards[i]
                    else:
                        self.y_[i] = self.epi_rewards[i] + self.discount * np.max(self.Q_tilde[i])
            
            self.batch_count = 1

            batch_exp = {
                "state": batch_states(self.epi_state[0:self.minibatch_size], self.device, self.phi),
                "action": torch.as_tensor(
                            self.epi_actions[0:self.minibatch_size], device=self.device
                        ),
                "target": torch.as_tensor(
                            self.y_[0:self.minibatch_size], device=self.device
                        ),
            }
            return self.update(batch_exp)
        # if an episode is still being updated, use the next minibatch of the already generated target value.
        else:
            self.batch_count += 1
            batch_exp = {
                "state": batch_states(self.epi_state[(self.batch_count-1)*self.minibatch_size:self.batch_count*self.minibatch_size], self.device, self.phi),
                "action": torch.as_tensor(
                            self.epi_actions[(self.batch_count-1)*self.minibatch_size:self.batch_count*self.minibatch_size], device=self.device
                        ),
                "target": torch.as_tensor(
                            self.y_[(self.batch_count-1)*self.minibatch_size:self.batch_count*self.minibatch_size], device=self.device
                        ),
            }
            return self.update(batch_exp)

    def update_if_necessary(self, iteration):
        """Update the model if the condition is met.

        Args:
            iteration (int): Timestep.

        Returns:
            bool: True iff the condition was updated this time.
        """
        if len(self.replay_buffer) < self.replay_start_size:
            return False

        if iteration % self.update_interval != 0:
            return False

        for _ in range(self.n_times_update):
            self._do_training()

        return True

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):

        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                    'reset': batch_reset[i]
                }
                if self.recurrent:
                    transition["recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states, i, detach=True
                        )
                    )
                    transition["next_recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states, i, detach=True
                        )
                    )
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.update_if_necessary(self.t)
            
            if self.replay_buffer_snapshot_path is not None and self.t % self.replay_buffer_snapshot_interval == 0:
                save_path = os.path.join(self.replay_buffer_snapshot_path, 'latest_replay_buffer.pkl')
                logging.info('Save replay buffer snapshot at {}'.format(save_path))
                with open(save_path, 'wb') as f:
                    dill.dump(self.replay_buffer, f) 
                
        if self.recurrent:
            # Reset recurrent states when episodes end
            self.train_prev_recurrent_states = None
            self.train_recurrent_states = _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                batch_done=batch_done,
                batch_reset=batch_reset,
                recurrent_states=self.train_recurrent_states,
            )

    def _can_start_replay(self):
        if len(self.replay_buffer) < self.replay_start_size:
            return False
        if self.recurrent and self.replay_buffer.n_episodes < self.minibatch_size:
            return False
        return True
