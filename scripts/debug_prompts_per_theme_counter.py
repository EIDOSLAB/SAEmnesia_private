#!/usr/bin/env python
"""
Load the style latent dictionary and print how many prompts we have for each theme.
"""

import os
import sys
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser(description="Count prompts per theme from style latent dictionary")
    parser.add_argument("--latent_file", type=str, required=True, 
                        help="Path to the style_latents_dict_{hookpoint}.pkl file")
    args = parser.parse_args()
    
    # Load the style latent dictionary
    print(f"Loading style latent dictionary from {args.latent_file}")
    try:
        with open(args.latent_file, "rb") as f:
            style_latents_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.latent_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Print the number of prompts for each theme
    print("\nPrompt count per theme:")
    print("-" * 30)
    for theme, latents in style_latents_dict.items():
        num_prompts = latents.shape[0]
        timesteps = latents.shape[1]
        latents = latents.shape[2]
        print(f"{theme}: {num_prompts} prompts, {timesteps} timesteps, {latents} latents")
    
    # Print total number of prompts
    total_prompts = sum(latents.shape[0] for latents in style_latents_dict.values())
    print("-" * 30)
    print(f"Total: {total_prompts} prompts")

if __name__ == "__main__":
    main()