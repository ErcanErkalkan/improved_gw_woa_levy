from PIL import Image

# Load the existing tall SOC comparison figure
img_path = "soc_comparison_subplots.png"
img = Image.open(img_path)

# Compute mid-point along height to split into top and bottom halves
width, height = img.size
mid_y = height // 2

# Crop top half and bottom half
top_half = img.crop((0, 0, width, mid_y-450))
bottom_half = img.crop((0, mid_y-450, width, height))

# Save the two halves
top_path = "soc_comparison_top.png"
bottom_path = "soc_comparison_bottom.png"
top_half.save(top_path)
bottom_half.save(bottom_path)

# Provide filenames for user reference
(top_path, bottom_path)
