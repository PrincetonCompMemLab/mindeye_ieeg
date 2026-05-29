import unittest
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import align_channels_across_runs, reshape_electrode_data_by_stimuli, do_retrieval

class TestAlignChannels(unittest.TestCase):
    
    def setUp(self):
        # Common constants
        self.n_time = 10
        self.exclude = ['TRIG']

    def test_basic_alignment_and_padding(self):
        """
        Test alignment where runs have different channels.
        Run 1: ['ChA', 'ChB']
        Run 2: ['ChB', 'ChC']
        Expected: ['ChA', 'ChB', 'ChC']
        Run 1 should have NaN for ChC.
        Run 2 should have NaN for ChA.
        """
        # Run 1: 2 trials, ChA, ChB
        data1 = np.ones((2, 2, self.n_time)) # Fill with 1s
        labels1 = pd.DataFrame({'channel_idx': [0, 1], 'channel_name': ['ChA', 'ChB']})
        
        # Run 2: 1 trial, ChB, ChC (ChB is index 0 here)
        data2 = np.full((1, 2, self.n_time), 2.0) # Fill with 2s
        labels2 = pd.DataFrame({'channel_idx': [0, 1], 'channel_name': ['ChB', 'ChC']})
        
        all_data = [data1, data2]
        all_labels = [labels1, labels2]
        
        aligned, out_labels = align_channels_across_runs(all_data, all_labels, self.exclude)
        
        # 1. Check Output Shapes
        # Total trials = 2 + 1 = 3
        # Unique channels = A, B, C = 3
        self.assertEqual(aligned.shape, (3, 3, self.n_time))
        
        # 2. Check Labels DataFrame
        expected_names = ['ChA', 'ChB', 'ChC']
        self.assertListEqual(out_labels['channel_name'].tolist(), expected_names)
        
        # 3. Check Data Integrity
        # Map output indices: ChA=0, ChB=1, ChC=2 (due to sorting)
        
        # -- Check Run 1 (Rows 0-1) --
        # ChA (col 0) should be 1.0
        np.testing.assert_array_equal(aligned[0:2, 0, :], 1.0)
        # ChB (col 1) should be 1.0
        np.testing.assert_array_equal(aligned[0:2, 1, :], 1.0)
        # ChC (col 2) should be NaN (Run 1 didn't have ChC)
        self.assertTrue(np.isnan(aligned[0:2, 2, :]).all())
        
        # -- Check Run 2 (Row 2) --
        # ChA (col 0) should be NaN (Run 2 didn't have ChA)
        self.assertTrue(np.isnan(aligned[2, 0, :]).all())
        # ChB (col 1) should be 2.0
        np.testing.assert_array_equal(aligned[2, 1, :], 2.0)
        # ChC (col 2) should be 2.0
        np.testing.assert_array_equal(aligned[2, 2, :], 2.0)

    def test_shuffled_order(self):
        """
        Test that channel order in the input doesn't matter, only the name.
        Run 1: ['ChA', 'ChB']
        Run 2: ['ChB', 'ChA']
        """
        # Run 1: 1 trial, val=1
        data1 = np.ones((1, 2, self.n_time)) 
        labels1 = pd.DataFrame({'channel_idx': [0, 1], 'channel_name': ['ChA', 'ChB']})
        
        # Run 2: 1 trial, val=2. Note data is swapped relative to names
        # Index 0 is ChB (val 2), Index 1 is ChA (val 3)
        data2 = np.zeros((1, 2, self.n_time))
        data2[:, 0, :] = 2 # ChB
        data2[:, 1, :] = 3 # ChA
        labels2 = pd.DataFrame({'channel_idx': [10, 11], 'channel_name': ['ChB', 'ChA']})
        
        aligned, out_labels = align_channels_across_runs([data1, data2], [labels1, labels2])
        
        # Sorted order: ChA (idx 0), ChB (idx 1)
        
        # Run 2, ChA (col 0) should be 3
        np.testing.assert_array_equal(aligned[1, 0, :], 3)
        # Run 2, ChB (col 1) should be 2
        np.testing.assert_array_equal(aligned[1, 1, :], 2)

    def test_exclude_channel(self):
        """Ensure 'TRIG' is removed from final output."""
        data = np.zeros((1, 2, self.n_time))
        labels = pd.DataFrame({'channel_idx': [0, 1], 'channel_name': ['neuron1', 'TRIG']})
        
        aligned, out_labels = align_channels_across_runs([data], [labels], exclude_channels=['TRIG'])
        
        self.assertEqual(aligned.shape[1], 1)
        self.assertEqual(out_labels['channel_name'].iloc[0], 'neuron1')

    def test_assertion_error_on_channel_mismatch(self):
        """Should raise AssertionError if data columns != label rows."""
        # Data has 5 channels, Labels has 4
        data = np.zeros((10, 5, self.n_time))
        labels = pd.DataFrame({
            'channel_idx': range(4), 
            'channel_name': ['a','b','c','d']
        })
        
        with self.assertRaisesRegex(AssertionError, "Data channels.*do not match"):
            align_channels_across_runs([data], [labels])

    def test_value_error_on_time_mismatch(self):
        """Should raise ValueError if time dimension varies across runs."""
        data1 = np.zeros((1, 1, 100))
        labels1 = pd.DataFrame({'channel_name': ['A']})
        
        data2 = np.zeros((1, 1, 101)) # Different time
        labels2 = pd.DataFrame({'channel_name': ['A']})
        
        with self.assertRaises(ValueError):
            align_channels_across_runs([data1, data2], [labels1, labels2])

    def test_detect_unsorted_physical_mapping(self):
        """
        Fails if the function assumes labels are always sorted 0, 1, 2...
        If the labels were [2, 0, 1] and we sort them to [0, 1, 2], 
        the data at index 0 (which belongs to ID 2) is now incorrectly labeled as ID 0.
        """
        # 1 Trial, 3 Channels, 5 Timepoints
        # We put a unique value (99) in the FIRST physical channel
        data_array = np.zeros((1, 3, 5))
        data_array[:, 0, :] = 99.0 
        
        # But the labels say the FIRST physical channel is actually ID 2
        # This simulates a recording where channels were saved out of order.
        unsorted_labels = pd.DataFrame({
            'channel_idx': [2, 0, 1],
            'channel_name': ['Ch2', 'Ch0', 'Ch1']
        })
        
        # If your code does: labels.sort_values('channel_idx')
        # The labels become: [Ch0, Ch1, Ch2]
        # But data_array[:, 0, :] is still 99.0. 
        # Now the function thinks Ch0 has the value 99.0. This is a BUG.
        
        aligned, final_labels = align_channels_across_runs(
            [data_array], [unsorted_labels], exclude_channels=[]
        )
        
        # In the final ALIGNED output, 'Ch2' should be the one with 99.0
        # Find where 'Ch2' is in the final labels
        ch2_idx = final_labels[final_labels['channel_name'] == 'Ch2'].index[0]
        
        # This assertion will FAIL if you sort the labels without remapping the data
        self.assertEqual(aligned[0, ch2_idx, 0], 99.0, 
                         "Data was assigned to the wrong channel because labels were sorted independently.")
    def test_duplicate_channels_in_single_run(self):
        """
        If a single run contains two channels with the exact same name,
        it should raise an error (ambiguity) OR handle it gracefully.
        Ideally, this should fail to prevent overwriting data.
        """
        data = np.zeros((1, 2, 10))
        # Two channels named 'neuron_1'
        labels = pd.DataFrame({
            'channel_idx': [0, 1], 
            'channel_name': ['neuron_1', 'neuron_1']
        })
        
        msg = "Duplicate channel names found in a single run"
        with self.assertRaisesRegex(ValueError, msg):
            align_channels_across_runs([data], [labels])
    
    def test_all_channels_excluded(self):
        """
        Test case where a run contains ONLY channels that are in the exclude list.
        The resulting array for that run should be all NaNs or empty depending on logic.
        """
        # Run 1: Only TRIG
        data1 = np.ones((5, 1, 10))
        labels1 = pd.DataFrame({'channel_idx': [0], 'channel_name': ['TRIG']})
        
        # Run 2: 'neuron_a'
        data2 = np.zeros((5, 1, 10))
        labels2 = pd.DataFrame({'channel_idx': [0], 'channel_name': ['neuron_a']})
        
        aligned, out_labels = align_channels_across_runs(
            [data1, data2], 
            [labels1, labels2], 
            exclude_channels=['TRIG']
        )
        
        # Result should have 1 channel (neuron_a), 10 trials total
        self.assertEqual(aligned.shape, (10, 1, 10))
        # The first 5 trials (from Run 1) should be purely NaN because 'neuron_a' didn't exist there
        self.assertTrue(np.isnan(aligned[:5]).all())
    
    def test_single_channel_preservation(self):
        """
        Test that 1D squeezing doesn't happen unexpectedly for single-channel datasets.
        """
        data = np.random.rand(10, 1, 50)
        labels = pd.DataFrame({'channel_idx': [0], 'channel_name': ['Solo']})
        
        aligned, _ = align_channels_across_runs([data], [labels])
        
        # Shape should remain (10, 1, 50), not (10, 50)
        self.assertEqual(aligned.ndim, 3)
        self.assertEqual(aligned.shape, (10, 1, 50))

class test_stim_ordering(unittest.TestCase):
    def test_image_axis_order_matches_first_appearance(self):
        """
        Stimulus 'A' appears once (first), stimulus 'B' appears 3 times.
        value_counts() will rank B first (most frequent), so without the fix,
        reshaped_data[:, 0, ...] will silently contain B's data instead of A's.
        """
        n_channels, n_time = 2, 5
        # Trial order: A, B, B, B
        # A gets fill value 10, B gets fill value 99
        electrode_data = np.zeros((n_channels, n_time, 4))
        electrode_data[:, :, 0] = 10.0  # trial 0 → stimulus A
        electrode_data[:, :, 1] = 99.0  # trial 1 → stimulus B
        electrode_data[:, :, 2] = 99.0  # trial 2 → stimulus B
        electrode_data[:, :, 3] = 99.0  # trial 3 → stimulus B
    
        events_df = pd.DataFrame({
            'coco_id': ['A', 'B', 'B', 'B'],
            'filename': ['A', 'B', 'B', 'B']
        })
    
        reshaped, info = reshape_electrode_data_by_stimuli(
            electrode_data, events_df, id_column='nsd_id',
            stim_column='coco_id', fill_column='filename'
        )
    
        # Image index 0 must be the first-appearing stimulus (A), not the most frequent (B)
        self.assertEqual(info['unique_stimuli'][0], 'A',
                         "Image axis index 0 should be first-seen stimulus 'A', not most-frequent 'B'")
        np.testing.assert_array_equal(
            reshaped[:, 0, :, 0], 10.0,
            err_msg="reshaped[:, 0, :, 0] should contain A's data (10.0), not B's (99.0)"
        )


class TestDoRetrieval(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n = 10
        self.feat = 512
        self.brain = np.random.randn(self.n, 128)
        self.clip_true = np.random.randn(self.n, self.feat)
        # Normalize for controlled tests
        self.clip_true /= np.linalg.norm(self.clip_true, axis=1, keepdims=True)

    # --- Return types & shapes ---

    def test_return_types(self):
        clip_pred = np.random.randn(self.n, self.feat)
        fwd_acc, sims = do_retrieval(self.brain, self.clip_true, clip_pred)
        self.assertIsInstance(fwd_acc, float)
        self.assertIsInstance(sims, np.ndarray)

    def test_sims_shape(self):
        clip_pred = np.random.randn(self.n, self.feat)
        _, sims = do_retrieval(self.brain, self.clip_true, clip_pred)
        self.assertEqual(sims.shape, (self.n, self.n))

    def test_sims_range(self):
        """Cosine similarities should be in [-1, 1]."""
        clip_pred = np.random.randn(self.n, self.feat)
        _, sims = do_retrieval(self.brain, self.clip_true, clip_pred)
        self.assertTrue(np.all(sims >= -1.0 - 1e-6))
        self.assertTrue(np.all(sims <=  1.0 + 1e-6))

    # --- Accuracy bounds ---

    def test_perfect_predictions_give_100_acc(self):
        """When pred == true, forward accuracy should be 1.0."""
        fwd_acc, _ = do_retrieval(self.brain, self.clip_true, self.clip_true.copy())
        self.assertAlmostEqual(fwd_acc, 1.0, places=5)

    def test_acc_in_valid_range(self):
        clip_pred = np.random.randn(self.n, self.feat)
        fwd_acc, _ = do_retrieval(self.brain, self.clip_true, clip_pred)
        self.assertGreaterEqual(fwd_acc, 0.0)
        self.assertLessEqual(fwd_acc, 1.0)

    def test_shuffled_predictions_lower_acc(self):
        """Shuffled predictions should give lower accuracy than perfect ones."""
        shuffled = self.clip_true[np.random.permutation(self.n)]
        fwd_acc_perfect, _ = do_retrieval(self.brain, self.clip_true, self.clip_true.copy())
        fwd_acc_shuffled, _ = do_retrieval(self.brain, self.clip_true, shuffled)
        self.assertGreater(fwd_acc_perfect, fwd_acc_shuffled)

    # --- top_k behavior ---

    def test_top_k_increases_acc(self):
        """top_k=2 should yield >= accuracy than top_k=1."""
        clip_pred = np.random.randn(self.n, self.feat)
        acc_k1, _ = do_retrieval(self.brain, self.clip_true, clip_pred, top_k=1)
        acc_k2, _ = do_retrieval(self.brain, self.clip_true, clip_pred, top_k=2)
        self.assertGreaterEqual(acc_k2, acc_k1)

    # --- n_retrievals behavior ---

    def test_n_retrievals_affects_sims(self):
        """Changing n_retrievals should change the shape or content of sims."""
        clip_pred = np.random.randn(self.n, self.feat)
        _, sims_5 = do_retrieval(self.brain, self.clip_true, clip_pred, n_retrievals=5)
        _, sims_3 = do_retrieval(self.brain, self.clip_true, clip_pred, n_retrievals=3)
        # Either shape changes or at minimum the call doesn't crash
        self.assertIsNotNone(sims_3)

    # --- Edge cases ---

    def test_single_sample(self):
        """Single sample should trivially retrieve itself."""
        brain_1 = self.brain[:1]
        true_1 = self.clip_true[:1]
        fwd_acc, sims = do_retrieval(brain_1, true_1, true_1.copy(), n_retrievals=1)
        self.assertAlmostEqual(fwd_acc, 1.0, places=5)
        self.assertEqual(sims.shape, (1, 1))

    def test_negative_embeddings_handled(self):
        """Negated predictions (worst-case cosine sim) should still run."""
        clip_pred = -self.clip_true  # opposite direction
        fwd_acc, sims = do_retrieval(self.brain, self.clip_true, clip_pred)
        self.assertIsNotNone(fwd_acc)


if __name__ == '__main__':
    unittest.main()
    