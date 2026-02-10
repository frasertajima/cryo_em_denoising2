# CRITICAL FINDING: Forward Pass Produces Wrong Results

## The Mystery

With a 1→1 conv layer, constant input=0.5, we observe:

### Step 1→2:
- Input: 0.5 (constant, verified)
- Weight mean: -0.00330 → -0.00278 (INCREASED by +0.00052)
- Bias: 0.000 → 0.00105 (INCREASED by +0.00105)
- **Output mean: -0.0232 → -0.0314 (DECREASED by -0.0082)**

### Step 2→3:
- Input: 0.5 (constant, verified)
- Weight mean: -0.00278 → -0.00226 (INCREASED by +0.00052)
- Bias: 0.00105 → 0.00211 (INCREASED by +0.00106)
- **Output mean: -0.0314 → -0.0364 (DECREASED by -0.0050)**

## Why This Is Impossible

For a convolution: `output = sum(weights * input) + bias`

With constant positive input and both weights and bias increasing, output MUST increase.
But it's decreasing!

## Verified Facts

1. ✓ Input is constant at 0.5 across all steps
2. ✓ Weights are increasing (less negative)
3. ✓ Bias is increasing  
4. ✓ grad_output is not modified by backward pass
5. ✓ Device synchronization doesn't help
6. ✓ Weight updates are mathematically correct (w = w - lr * grad)
7. ✓ One-step convergence test works perfectly
8. ✗ Multi-step training produces wrong forward pass outputs

## Hypotheses to Test

1. Forward pass is reading stale weights from device?
2. cuDNN has internal state that's being corrupted?
3. Workspace memory is being reused incorrectly?
4. There's a bug in how weights are stored/retrieved?

## Next Steps

Need to isolate whether this is:
- A cuDNN API usage bug
- A Fortran array layout issue  
- A device memory coherency problem
