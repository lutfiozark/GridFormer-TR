import os
import torch
import argparse
from torch.autograd import gradcheck

# Add the project root to sys.path
import sys
sys.path.append(os.getcwd())

try:
    from basicsr.losses.dists_loss import DISTS
except ImportError:
    # Try registering directly
    print("Could not import DISTS directly, trying from basicsr.losses")
    from basicsr.losses import DISTS


def test_dists_loss():
    print("Testing DISTS loss implementation...")

    # Create test tensors (3x64x64 images) in range [0, 1]
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)

    # Initialize DISTS loss
    dists_loss = DISTS(loss_weight=1.0, reduction='mean', calibrated=True)

    # Test forward pass
    try:
        loss = dists_loss(pred, target)
        print(f"Forward pass successful, loss value: {loss.item()}")

        # Test backward pass
        loss.backward()
        if pred.grad is not None:
            print(
                f"Backward pass successful, gradient shape: {pred.grad.shape}")
            grad_mean = pred.grad.abs().mean().item()
            print(f"Gradient magnitude (mean abs): {grad_mean:.6f}")
        else:
            print("Backward pass failed: No gradients computed")

        # Check that alpha and beta are loaded correctly
        print(f"Alpha values: {dists_loss.alpha}")
        print(f"Beta values: {dists_loss.beta}")

        # Run with non-default parameters
        dists_loss2 = DISTS(loss_weight=0.5, reduction='sum', calibrated=False)
        loss2 = dists_loss2(pred, target)
        print(f"Loss with different parameters, value: {loss2.item()}")

        # Check uncalibrated alpha and beta
        print(f"Uncalibrated alpha values: {dists_loss2.alpha}")
        print(f"Uncalibrated beta values: {dists_loss2.beta}")

        print("\nDISTS loss test completed successfully!")

    except Exception as e:
        print(f"Error testing DISTS loss: {e}")
        import traceback
        traceback.print_exc()


def test_numerical_gradient():
    print("\nTesting numerical gradient stability (this may take a while)...")

    # Use smaller images for gradient check (expensive operation)
    pred = torch.rand(2, 3, 32, 32, requires_grad=True, dtype=torch.double)
    target = torch.rand(2, 3, 32, 32, dtype=torch.double)

    # Move to CUDA if available
    if torch.cuda.is_available():
        print("Using CUDA for gradient check")
        pred = pred.cuda()
        target = target.cuda()

    # Use a simplified version for gradient check
    dists_loss = DISTS(loss_weight=1.0, reduction='sum', calibrated=True)

    try:
        # Test if gradients are numerically correct using PyTorch's gradcheck
        result = gradcheck(lambda x: dists_loss(
            x, target), pred, eps=1e-4, atol=1e-3)
        print(f"Numerical gradient check {'passed' if result else 'failed'}")
    except Exception as e:
        print(f"Error in numerical gradient check: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_grad', action='store_true',
                        help='Run numerical gradient test (slow)')
    args = parser.parse_args()

    test_dists_loss()

    if args.test_grad:
        test_numerical_gradient()
