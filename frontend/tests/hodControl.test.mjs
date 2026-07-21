import test from 'node:test';
import assert from 'node:assert/strict';
import { coverageLabel, coverageTone, filterCoverageRows, sourceValue } from '../src/utils/hodControl.js';

const rows = [
  { roll: '24011CSEAI0110', name: 'SURYA', subjects: ['CVO'], coverage_status: 'ready' },
  { roll: '2401100CSE0268', name: 'NITHIN', subjects: ['CVO'], coverage_status: 'missing_both' },
  { roll: '2401100CSE0006', name: 'YAAMINI', subjects: ['SWE', 'CCM'], coverage_status: 'dataset_only' },
];

test('coverage labels and tones are explicit', () => {
  assert.equal(coverageLabel('ready'), 'Ready');
  assert.equal(coverageLabel('missing_both'), 'Missing dataset & embedding');
  assert.equal(coverageTone('ready'), 'success');
  assert.equal(coverageTone('dataset_only'), 'warning');
  assert.equal(sourceValue(null), 'Unknown');
});

test('coverage filters combine subject, search, and issue state', () => {
  assert.deepEqual(filterCoverageRows(rows, { subject: 'CVO', issuesOnly: true }).map((row) => row.roll), ['2401100CSE0268']);
  assert.deepEqual(filterCoverageRows(rows, { search: 'yaa' }).map((row) => row.roll), ['2401100CSE0006']);
  assert.equal(filterCoverageRows(rows, { subject: 'SWE' }).length, 1);
});

test('AI0110 and CSE0110 remain exact strings without aliasing', () => {
  const distinct = [
    { roll: '24011CSEAI0110', subjects: ['CVO'], coverage_status: 'ready' },
    { roll: '2401100CSE0110', subjects: ['CVO'], coverage_status: 'missing_both' },
  ];
  assert.equal(filterCoverageRows(distinct, { search: '24011CSEAI0110' }).length, 1);
  assert.equal(filterCoverageRows(distinct, { search: '2401100CSE0110' }).length, 1);
});
