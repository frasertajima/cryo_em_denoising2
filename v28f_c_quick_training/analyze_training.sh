#!/bin/bash
# Run the simple test and capture detailed output
./simple_overfit_test 2>&1 | tee training_output.log

# Extract key metrics
echo ""
echo "=== ANALYSIS ==="
grep "Step.*diagnostics" training_output.log -A 7
