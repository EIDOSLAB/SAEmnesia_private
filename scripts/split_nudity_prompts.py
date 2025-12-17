# split_prompts.py
input_file = "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/annotations/coco_30k_captions.txt"
nudity_output = "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/annotations/nudity_prompts.txt"
non_nudity_output = "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/annotations/non_nudity_prompts.txt"

# Define nudity-related keywords (you can expand this list)
nudity_keywords = ['naked', 'nude', 'nudity', 'unclothed', 'bare']

with open(input_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

nudity_prompts = []
non_nudity_prompts = []

for line in lines:
    line_lower = line.lower()
    if any(keyword in line_lower for keyword in nudity_keywords):
        nudity_prompts.append(line)
    else:
        non_nudity_prompts.append(line)

# Save to separate files
with open(nudity_output, 'w') as f:
    f.write('\n'.join(nudity_prompts))

with open(non_nudity_output, 'w') as f:
    f.write('\n'.join(non_nudity_prompts))

print(f"Nudity prompts: {len(nudity_prompts)}")
print(f"Non-nudity prompts: {len(non_nudity_prompts)}")