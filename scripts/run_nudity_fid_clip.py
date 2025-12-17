import subprocess
import fire
import numpy as np
import torch
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def run_evaluation_for_all(
    reference_dir,
    generated_dir,
    output_dir,
    compute_fid=True,
    compute_clip=False,
    captions_file=None,
    batch_size=64
):
    """
    Run FID/CLIP evaluation for generated images
    
    Args:
        reference_dir: Directory containing reference images
        generated_dir: Directory containing generated images
        output_dir: Directory to save results
        compute_fid: Whether to compute FID
        compute_clip: Whether to compute CLIP score
        captions_file: Path to captions JSON file (for CLIP)
        batch_size: Batch size for processing
    """
    
    # Build command
    command = (
        f"PYTHONPATH=. python scripts/fid_clip_evaluation.py "
        f"--generated_dir '{generated_dir}' "
        f"--reference_dir '{reference_dir}' "
        f"--output_dir '{output_dir}' "
        f"--batch_size {batch_size} "
    )
    
    if compute_fid:
        command += "--compute_fid "
    
    if compute_clip:
        command += "--compute_clip "
        if captions_file:
            command += f"--captions_file '{captions_file}' "
    
    print(f"Running command: {command}")
    process = subprocess.run(command, shell=True)
    
    if process.returncode != 0:
        print(f"Error: Script failed with return code {process.returncode}")
        return None
    else:
        print(f"Successfully completed evaluation")
        
        # Load and return results
        results = {}
        if compute_fid:
            fid_path = os.path.join(output_dir, "fid_score.pth")
            if os.path.exists(fid_path):
                results['fid'] = torch.load(fid_path)
        
        if compute_clip:
            clip_path = os.path.join(output_dir, "clip_scores.pth")
            if os.path.exists(clip_path):
                clip_data = torch.load(clip_path)
                results['clip_score'] = clip_data['clip_score']
        
        return results


def main(
    reference_dir,
    generated_dir,
    output_dir,
    compute_fid=True,
    compute_clip=False,
    captions_file=None,
    batch_size=64
):
    """
    Main function to run FID/CLIP evaluation
    
    Args:
        reference_dir: Directory with reference images
        generated_dir: Directory with generated images  
        output_dir: Directory to save results
        compute_fid: Compute FID score
        compute_clip: Compute CLIP score
        captions_file: Path to captions file for CLIP
        batch_size: Batch size for processing
    """
    
    print(f"\n{'='*70}")
    print("Starting FID/CLIP Evaluation")
    print(f"{'='*70}")
    print(f"Reference: {reference_dir}")
    print(f"Generated: {generated_dir}")
    print(f"Output: {output_dir}")
    print(f"FID: {compute_fid}, CLIP: {compute_clip}")
    print(f"{'='*70}\n")
    
    results = run_evaluation_for_all(
        reference_dir=reference_dir,
        generated_dir=generated_dir,
        output_dir=output_dir,
        compute_fid=compute_fid,
        compute_clip=compute_clip,
        captions_file=captions_file,
        batch_size=batch_size
    )
    
    if results:
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        if 'fid' in results:
            print(f"FID Score: {results['fid']:.4f}")
        if 'clip_score' in results:
            print(f"CLIP Score: {results['clip_score']:.4f}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    fire.Fire(main)