"""Check the numerical and reference files without downloading ECG-ID."""
from collections import defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_generated_manuscript_tables_match_saved_metrics(self):
        spec = importlib.util.spec_from_file_location(
            "paper_tables", ROOT / "research/build_paper_tables.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, expected in module.generated_tables().items():
            with self.subTest(table=name):
                self.assertEqual((ROOT / "paper/tables" / name).read_text(encoding="utf-8"), expected)
        manuscript = (ROOT / 'paper/main.tex').read_text(encoding='utf-8')
        self.assertEqual(manuscript, module.embed_tables(manuscript, module.generated_tables()))
        self.assertNotIn(r'\input{tables/', manuscript)

    def test_historical_metrics_are_complete_and_aggregate_consistently(self):
        for run, people in (("1_same_recording_gaussian", 30),
                            ("2_new_recording_gaussian", 10),
                            ("3_new_recording_morlet", 10)):
            folder = ROOT / "results" / run / "metrics"
            groups = defaultdict(dict)
            with (folder / "per_subject_long.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row['feature_set'], row['model'], row['metric'])
                    subject, value = int(row['subject']), float(row['value'])
                    self.assertNotIn(subject, groups[key])
                    self.assertTrue(math.isfinite(value) and 0 <= value <= 1)
                    groups[key][subject] = value
            self.assertEqual(len(groups), 2 * 6 * 3)
            self.assertTrue(all(len(values) == people for values in groups.values()))
            with (folder / "benchmark_results.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    fk = 'raw' if row['Feature set'].startswith('Raw') else 'wtmm'
                    values = [round(x, 4) for x in groups[fk, row['Model'], row['Metric'].lower()].values()]
                    self.assertAlmostEqual(statistics.mean(values), float(row['Mean']), delta=0.00011)
                    self.assertAlmostEqual(statistics.stdev(values), float(row['Std']), delta=0.00011)

    def test_manuscript_citations_and_cross_references_resolve(self):
        tex = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
        bibliography = re.findall(r"\\bibitem\{([^}]+)\}", tex)
        self.assertEqual(len(bibliography), len(set(bibliography)))
        cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex)
                 for key in group.split(',')}
        self.assertFalse(cited - set(bibliography))
        labels = re.findall(r"\\label\{([^}]+)\}", tex)
        self.assertEqual(len(labels), len(set(labels)))
        references = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex))
        self.assertFalse(references - set(labels))

    def test_notebook_code_cells_compile(self):
        for path in (ROOT / "notebook").glob('*.ipynb'):
            notebook = json.loads(path.read_text(encoding='utf-8'))
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'{path.name}:cell{index}', 'exec')


if __name__ == "__main__":
    unittest.main()
