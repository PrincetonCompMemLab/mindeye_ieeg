import unittest
import numpy as np
import pandas as pd

from utils import align_channels_across_runs

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

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)