import json
import os, sys
import fire
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from UnlearnCanvas_resources.const import class_available, theme_available

def safe_load_json(json_path):
    """Safely load JSON with error handling for corrupted files"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        # Try to salvage the file by truncating to last complete JSON object
        try:
            with open(json_path, 'r') as f:
                content = f.read()
            
            # Find the last complete '}' 
            last_brace = content.rfind('}')
            if last_brace != -1:
                truncated_content = content[:last_brace + 1]
                return json.loads(truncated_content)
        except json.JSONDecodeError:
            pass
        
        # If salvage fails, return None
        return None
    except Exception as e:
        # Handle other errors (file not readable, etc.)
        return None

def main(input_dir: str, attk_idxs: list[int]):
    total_success_before_attack = 0
    total_success_after_attack = 0
    total_attempts = 0
    skipped_files = 0
    corrupted_files = []
    missing_files = 0
    
    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    
    # Calculate total expected files for progress tracking
    total_expected = len(attk_idxs) * len(class_available) * len(theme_avail)
    
    progress_bar = tqdm(total=total_expected)
    
    for cls in class_available:
        for theme in theme_avail:
            for attk_idx in attk_idxs:
                json_path = os.path.join(
                    input_dir, cls, theme, f"attack_idx_{attk_idx}", "log.json"
                )
                
                # Check if file exists
                if not os.path.exists(json_path):
                    missing_files += 1
                    progress_bar.update(1)
                    continue
                
                # Try to load JSON safely
                json_log = safe_load_json(json_path)
                
                if json_log is None:
                    # File is corrupted and couldn't be salvaged
                    skipped_files += 1
                    corrupted_files.append(json_path)
                    progress_bar.update(1)
                    continue
                
                # Validate that we have the expected structure
                try:
                    if not isinstance(json_log, list) or len(json_log) == 0:
                        skipped_files += 1
                        corrupted_files.append(f"{json_path} (invalid structure - not a list or empty)")
                        progress_bar.update(1)
                        continue
                    
                    # Check if first and last elements have 'success' key
                    if 'success' not in json_log[0] or 'success' not in json_log[-1]:
                        skipped_files += 1
                        corrupted_files.append(f"{json_path} (missing 'success' key)")
                        progress_bar.update(1)
                        continue
                    
                    # Process valid data
                    total_success_after_attack += json_log[-1]["success"]
                    total_success_before_attack += json_log[0]["success"]
                    total_attempts += 1
                    
                except (KeyError, IndexError, TypeError) as e:
                    skipped_files += 1
                    corrupted_files.append(f"{json_path} (data structure error: {e})")
                    
                progress_bar.update(1)
                
                # Update progress description
                if total_attempts > 0:
                    ua_before = ((total_attempts - total_success_before_attack) / total_attempts) * 100
                    ua_after = ((total_attempts - total_success_after_attack) / total_attempts) * 100
                    progress_bar.set_description(
                        f"UA before: {ua_before:.2f}%, UA after: {ua_after:.2f}% | Skipped: {skipped_files}"
                    )
    
    progress_bar.close()
    
    # Final results
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total expected files: {total_expected}")
    print(f"Missing files: {missing_files}")
    print(f"Corrupted/skipped files: {skipped_files}")
    print(f"Successfully processed files: {total_attempts}")
    print(f"Success rate: {(total_attempts / total_expected) * 100:.2f}%")
    
    if total_attempts > 0:
        ua_before = ((total_attempts - total_success_before_attack) / total_attempts) * 100
        ua_after = ((total_attempts - total_success_after_attack) / total_attempts) * 100
        
        print("\n" + "="*80)
        print("ATTACK RESULTS")
        print("="*80)
        print(f"Average UA before attack: {ua_before:.2f}%")
        print(f"Average UA after attack: {ua_after:.2f}%")
        print(f"Attack effectiveness: {ua_before - ua_after:.2f} percentage points reduction")
    else:
        print("\nNo valid files found to process!")
    
    # Report on corrupted files
    if skipped_files > 0:
        print(f"\n" + "="*80)
        print("CORRUPTED FILES REPORT")
        print("="*80)
        print(f"Total corrupted files: {skipped_files}")
        
        if skipped_files <= 20:  # Show details if not too many
            print("\nCorrupted files list:")
            for i, corrupted_file in enumerate(corrupted_files, 1):
                print(f"{i:3d}. {corrupted_file}")
        else:
            print(f"\nToo many corrupted files to list ({skipped_files} total)")
            print("First 10 corrupted files:")
            for i, corrupted_file in enumerate(corrupted_files[:10], 1):
                print(f"{i:3d}. {corrupted_file}")
            print("...")
        
        # Recommendation
        corruption_rate = (skipped_files / total_expected) * 100
        print(f"\nCorruption rate: {corruption_rate:.2f}%")
        
        if corruption_rate > 10:
            print("⚠️  HIGH CORRUPTION RATE! Consider regenerating the corrupted files.")
        elif corruption_rate > 5:
            print("⚠️  MODERATE CORRUPTION RATE. You may want to regenerate some files.")
        else:
            print("✅ LOW CORRUPTION RATE. Should be fine to proceed with current data.")

if __name__ == "__main__":
    fire.Fire(main)