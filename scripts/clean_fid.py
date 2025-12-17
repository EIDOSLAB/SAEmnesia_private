from cleanfid import fid
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_dir', required=True)
    parser.add_argument('--reference_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    print(f"Computing FID using clean-fid library...")
    print(f"Generated: {args.generated_dir}")
    print(f"Reference: {args.reference_dir}")
    
    score = fid.compute_fid(
        args.generated_dir,
        args.reference_dir,
        mode="clean",
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"\nFID Score: {score:.2f}")

if __name__ == "__main__":
    main()
