import gdown
import os
import time
from urllib.parse import urlparse, parse_qs

def extract_folder_id(url):
    """Extract folder ID from Google Drive URL"""
    if '/folders/' in url:
        return url.split('/folders/')[1].split('?')[0]
    elif 'id=' in url:
        parsed = urlparse(url)
        return parse_qs(parsed.query)['id'][0]
    else:
        return url

def download_folder_safe(folder_url, output_path, max_retries=3):
    """Download a folder with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"  Downloading to: {output_path}")
            gdown.download_folder(
                folder_url,
                output=output_path,
                quiet=False,
                remaining_ok=True,
                use_cookies=False
            )
            return True
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  Failed to download {folder_url} after {max_retries} attempts")
                return False

def download_ucanvas_dataset(base_folder_id, output_dir="./ucanvas_dataset"):
    """
    Download the complete UCanvas dataset
    
    Args:
        base_folder_id: The main folder ID containing all styles
        output_dir: Local directory to save the dataset
    """
    
    # Available themes and classes
    themes = [
        "Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine",
        "Cartoon", "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism",
        "Dadaism", "Dapple", "Defoliation", "Early_Autumn", "Expressionism", "Fauvism",
        "French", "Glowing_Sunset", "Gorgeous_Love", "Greenfield", "Impressionism",
        "Ink_Art", "Joy", "Liquid_Dreams", "Magic_Cube", "Meta_Physics", "Meteor_Shower",
        "Monet", "Mosaic", "Neon_Lines", "On_Fire", "Pastel", "Pencil_Drawing", "Picasso",
        "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images", "Sketch", "Sponge_Dabbed",
        "Structuralism", "Superstring", "Surrealism", "Ukiyoe", "Van_Gogh", "Vibrant_Flow",
        "Warm_Love", "Warm_Smear", "Watercolor", "Winter"
    ]
    
    classes = [
        "Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes",
        "Flame", "Flowers", "Frogs", "Horses", "Human", "Jellyfish", "Rabbits",
        "Sandwiches", "Sea", "Statues", "Towers", "Trees", "Waterfalls"
    ]
    
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # First, try to download the entire folder structure at once
    base_folder_url = f"https://drive.google.com/drive/folders/{base_folder_id}"
    print(f"Attempting to download entire dataset from: {base_folder_url}")
    print(f"Output directory: {output_dir}")
    
    success = download_folder_safe(base_folder_url, output_dir)
    
    if success:
        # Check what we got
        downloaded_items = []
        for root, dirs, files in os.walk(output_dir):
            downloaded_items.extend(files)
        
        print(f"\n✅ Successfully downloaded {len(downloaded_items)} files!")
        
        # Show structure
        print("\nDataset structure:")
        for theme in themes[:5]:  # Show first 5 themes
            theme_path = os.path.join(output_dir, theme)
            if os.path.exists(theme_path):
                class_count = len([d for d in os.listdir(theme_path) if os.path.isdir(os.path.join(theme_path, d))])
                print(f"  {theme}: {class_count} object classes")
        
        if len(themes) > 5:
            print(f"  ... and {len(themes) - 5} more themes")
            
    else:
        print("❌ Failed to download the complete dataset")
        print("\nThis might be due to:")
        print("1. The folder being private (requires authentication)")
        print("2. Network issues")
        print("3. Google Drive API limits")
        print("\nFor large datasets, consider using rclone instead:")
        print("  rclone config  # Configure Google Drive access")
        print(f"  rclone copy 'gdrive:folder_name' {output_dir}/ -P")

def main():
    # Your main folder ID
    main_folder_id = "1-1Sc8h_tGArZv5Y201ugTF0K0D_Xn2lM"
    
    print("🎨 UCanvas Dataset Downloader")
    print("=" * 50)
    print(f"Downloading from folder ID: {main_folder_id}")
    
    download_ucanvas_dataset(main_folder_id)
    
    print("\n✨ Download process completed!")

if __name__ == "__main__":
    main()