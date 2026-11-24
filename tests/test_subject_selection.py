import contextlib
import io
import tempfile
import unittest
import warnings
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

import CT_MIFNet as bci_2a
import CT_MIFNet_2b as bci_2b
from commom_spatial_pattern_newdata import csp as edlep_csp


class SubjectSelectionTests(unittest.TestCase):
    def test_bci_entry_points_read_each_requested_subject_in_order(self):
        subjects = (7, 2)
        for module, channels, classes in ((bci_2a, 22, 4), (bci_2b, 3, 2)):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory(prefix="BCI data ") as root:
                labels = np.repeat(np.arange(1, classes + 1), 3)[:, None]
                rng = np.random.default_rng(11)
                for subject in subjects:
                    if module is bci_2a:
                        stem = f"A{subject:02d}"
                    else:
                        # Exercise the MATLAB and legacy naming conventions in one run.
                        prefix = "B" if subject == 7 else "A"
                        stem = f"{prefix}{subject:02d}processing_"
                    for split in ("T", "E"):
                        savemat(Path(root) / f"{stem}{split}.mat", {
                            "data": rng.standard_normal((64, channels, len(labels))),
                            "label": labels,
                            "marker": f"{subject}{split}",
                        })

                seen = []

                def read_subject(trainer):
                    train, _, evaluation, _ = trainer.get_source_data()
                    self.assertEqual(train.shape[0], len(labels))
                    self.assertEqual(evaluation.shape[0], len(labels))
                    seen.append((
                        trainer.nSub,
                        trainer.total_data["marker"].item(),
                        trainer.test_tmp["marker"].item(),
                    ))
                    return 0.5, 0.5, None, None, 0.0

                with patch.object(module, "CT_MIFNet"), patch.object(module.nn, "DataParallel"), \
                     patch.object(module.Trans, "train", read_subject), contextlib.redirect_stdout(io.StringIO()):
                    module.main(subjects=subjects, data_dir=root)

                self.assertEqual(seen, [(7, "7T", "7E"), (2, "2T", "2E")])

    def test_2b_selects_a_complete_pair(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root)
            b_train, b_eval = data / "B04processing_T.mat", data / "B04processing_E.mat"
            a_train, a_eval = data / "A04processing_T.mat", data / "A04processing_E.mat"
            b_train.touch()
            a_eval.touch()
            with self.assertRaisesRegex(FileNotFoundError, "B04processing_E.mat"):
                bci_2b._subject_files(data, 4)

            a_train.touch()
            self.assertEqual(bci_2b._subject_files(data, 4), (a_train, a_eval))
            b_eval.touch()
            self.assertEqual(bci_2b._subject_files(data, 4), (b_train, b_eval))

    def test_binary_csp_keeps_the_columns_for_absent_classes(self):
        rng = np.random.default_rng(31)
        for csp, channels in ((bci_2b.csp, 3), (edlep_csp, 32)):
            with self.subTest(channels=channels), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                data = rng.standard_normal((6, 64, channels))
                labels = np.array([0, 1, 1, 1, 1, 1])
                filters = csp(data, labels)
                self.assertEqual(filters.shape, (channels, 8))
                self.assertTrue(np.isfinite(filters).all())
                np.testing.assert_array_equal(filters[:, 4:], np.zeros((channels, 4)))

    @unittest.skipUnless(find_spec("mne"), "EDLEP uses MNE-Python")
    def test_edlep_log_records_the_selected_ids(self):
        import CT_MIFNet_newdata as edlep

        result = ([0.8], [0.7], [], [], [0.5], 0.1)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "results"
            with patch.object(edlep, "TFCformer"), patch.object(edlep.nn, "DataParallel"), \
                 patch.object(edlep.Trans, "train", return_value=result), contextlib.redirect_stdout(io.StringIO()):
                edlep.main(subjects=(8, 10), data_dir=root, output_dir=output)

            lines = (output / "TFCformer_best_acc.txt").read_text().splitlines()
            self.assertEqual([line.split(":", 1)[0] for line in lines], ["Subject8", "Subject10"])
