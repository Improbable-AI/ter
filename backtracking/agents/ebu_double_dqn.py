from backtracking.agents.dqn import DQN
from pfrl.utils import evaluating
from pfrl.utils.recurrent import pack_and_forward

from .ebu_dqn import EBUDQN

class DoubleEBUDQN(EBUDQN):
    """Double DQN.

    See: http://arxiv.org/abs/1509.06461.
    """

    def _compute_t(self, states):

        batch_next_state = states

        with evaluating(self.model):
            if self.recurrent:
                next_qout, _ = pack_and_forward(
                    self.model, batch_next_state, exp_batch["next_recurrent_state"],
                )
            else:
                next_qout = self.model(batch_next_state)

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model, batch_next_state, exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)

        return target_next_qout.q_values