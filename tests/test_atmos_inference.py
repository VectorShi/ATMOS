"""Inference gradient and trajectory interval contracts, without model weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from atmos.model.atmos import AtmosProtenix
from atmos.runner.md_inference import AtmosInferenceRunner


class InferenceGradientTests(unittest.TestCase):
    def test_direct_runner_forward_disables_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        grad_states = []

        def forward(**kwargs):
            grad_states.append(torch.is_grad_enabled())
            return {"coordinate": parameter * 3}, kwargs["label_dict"], {}

        runner = AtmosInferenceRunner.__new__(AtmosInferenceRunner)
        runner.models = [forward]
        batch = {"input_feature_dict": {}, "label_dict": {}, "label_full_dict": {}}
        with torch.enable_grad():
            result, _ = runner.model_forward(batch)
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(grad_states, [False])
        self.assertFalse(result["pred_dict"]["coordinate"].requires_grad)
        self.assertIsNone(result["pred_dict"]["coordinate"].grad_fn)


class DestinationIntervalTests(unittest.TestCase):
    def make_rollout(self, intervals, destination=True, prefix=1, history=30):
        batch_size = len(intervals)
        model = AtmosProtenix.__new__(AtmosProtenix)
        torch.nn.Module.__init__(model)
        model.destination_enable = destination
        model.model_max_track_length = history
        model.overlap_structure_num = 2
        model.configs = SimpleNamespace(sample_diffusion={"N_sample": 1, "N_step": 1})
        model.get_dynamic_temperature = Mock(return_value=torch.zeros(1))
        state = (torch.zeros(batch_size, 2, 4), torch.zeros(batch_size, 2, 2, 4))
        model.run_dyna_ssm = Mock(return_value=state)
        model.run_dyna_ssm_one_cycle = Mock(return_value=state)
        model.prepare_diffusion_cache = Mock(return_value={})
        model.sample_dynamic_diffusion = Mock(
            return_value=torch.zeros(batch_size, 1, 2, 3)
        )
        features = {
            "frame_atom_index": torch.zeros(2, 3, dtype=torch.long),
            "pae_rep_atom_mask": torch.ones(2, dtype=torch.bool),
            "has_frame": torch.ones(2, dtype=torch.bool),
            "distogram_rep_atom_mask": torch.ones(2, dtype=torch.bool),
            "intervals": torch.tensor(intervals, dtype=torch.float32),
            "coordinate": torch.zeros(batch_size, prefix + int(destination), 2, 3),
            "coordinate_mask": torch.ones(batch_size, prefix + int(destination), 2),
        }
        args = {
            "input_feature_dict": features,
            "s_inputs": torch.zeros(2, 4),
            "s_init": torch.zeros(2, 4),
            "z_init": torch.zeros(2, 2, 4),
            "s": torch.zeros(2, 4),
            "z": torch.zeros(2, 2, 4),
            "infer_trajectory_max_length": len(intervals[0]) + (not destination),
        }
        return model, args

    def test_destination_rejects_nonunit_steps_before_sampling(self):
        for intervals in ([[20, 2, 2]], [[20, 1, 2]], [[20, 0, 1]], [[20, 0.5, 1]]):
            with self.subTest(intervals=intervals):
                model, args = self.make_rollout(intervals)
                with self.assertRaisesRegex(ValueError, "unit transition intervals"):
                    model.rollout_dynamic_trajectory(**args)
                model.run_dyna_ssm.assert_not_called()
                model.sample_dynamic_diffusion.assert_not_called()

    def test_checks_all_batch_members(self):
        model, args = self.make_rollout([[20, 1, 1], [20, 1, 2]])
        with self.assertRaisesRegex(ValueError, "unit transition intervals"):
            model.rollout_dynamic_trajectory(**args)

    def test_coarse_stage_accepts_nonunit_intervals(self):
        for interval in (10, 20):
            with self.subTest(interval=interval):
                model, args = self.make_rollout([[interval, interval]], destination=False)
                output = model.rollout_dynamic_trajectory(**args)
                self.assertEqual(output.shape, (1, 3, 2, 3))

    def test_destination_horizon_is_separate_from_transition_step(self):
        for horizon in (10, 20):
            with self.subTest(horizon=horizon):
                model, args = self.make_rollout([[horizon, 1, 1, 1]])
                output = model.rollout_dynamic_trajectory(**args)
                self.assertEqual(output.shape, (1, 4, 2, 3))
                calls = model.run_dyna_ssm_one_cycle.call_args_list
                self.assertEqual(
                    [c.kwargs["intervals"][0, 0].item() for c in calls],
                    [horizon - 1, horizon - 2],
                )

    def test_stitching_prefix_and_history_reset_keep_remaining_horizon(self):
        model, args = self.make_rollout([[20] + [1] * 8], prefix=5, history=6)
        output = model.rollout_dynamic_trajectory(**args)
        self.assertEqual(output.shape, (1, 9, 2, 3))
        calls = model.run_dyna_ssm.call_args_list
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(
            calls[1].kwargs["intervals"], torch.tensor([[16., 1., 1.]])
        )
        calls = model.run_dyna_ssm_one_cycle.call_args_list
        self.assertEqual([c.kwargs["intervals"][0, 0].item() for c in calls], [14, 13])


if __name__ == "__main__":
    unittest.main()
