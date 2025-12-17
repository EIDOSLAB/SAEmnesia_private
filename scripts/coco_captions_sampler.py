import json
import random

# Set seed for reproducibility
random.seed(42)

# Load the COCO captions
with open('/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/annotations/captions_val2014.json', 'r') as f:
    data = json.load(f)

# Extract all captions
all_captions = [item['caption'] for item in data['annotations']]

print(f"Total captions available: {len(all_captions)}")

# Randomly sample 30,000 captions
# sampled_captions = random.sample(all_captions, 30000)
sampled_captions = all_captions[:30000]

# Add the two additional prompts
# sampled_captions.append("naked man")
# sampled_captions.append("naked woman")

# Save to text file (one caption per line)
with open('/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/val_coco_30k_captions.txt', 'w') as f:
    for caption in sampled_captions:
        f.write(caption + '\n')

print(f"Saved {len(sampled_captions)} captions to coco_30k_captions.txt")
print(f"First 5 captions:")
for i, caption in enumerate(sampled_captions[:5], 1):
    print(f"{i}. {caption}")