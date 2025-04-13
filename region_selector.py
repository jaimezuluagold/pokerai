"""
Region Selector Tool for Poker AI

This tool allows you to visually select regions on the screen and get their coordinates.
"""

import cv2
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import messagebox, simpledialog

# Global variables
regions = {}
current_region = None
start_point = None
end_point = None
image = None

def take_screenshot():
    """Take a screenshot of the entire screen."""
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for region selection."""
    global start_point, end_point, image, current_region
    
    img_copy = image.copy()
    
    # Draw existing regions
    for name, coords in regions.items():
        cv2.rectangle(img_copy, 
                     (coords['left'], coords['top']), 
                     (coords['left'] + coords['width'], coords['top'] + coords['height']),
                     (0, 255, 0), 2)
        cv2.putText(img_copy, name, (coords['left'], coords['top'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        end_point = (x, y)
        # Draw rectangle
        cv2.rectangle(img_copy, start_point, end_point, (0, 0, 255), 2)
    
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        
        # Get region name
        root = tk.Tk()
        root.withdraw()
        region_name = simpledialog.askstring("Region Name", 
                                            "Enter a name for this region:",
                                            parent=root)
        root.destroy()
        
        if region_name:
            # Calculate coordinates
            left = min(start_point[0], end_point[0])
            top = min(start_point[1], end_point[1])
            width = abs(end_point[0] - start_point[0])
            height = abs(end_point[1] - start_point[1])
            
            # Store region
            regions[region_name] = {
                'left': left,
                'top': top,
                'width': width,
                'height': height
            }
            
            print(f"Region '{region_name}' added: left={left}, top={top}, width={width}, height={height}")
            
            # Draw final rectangle
            cv2.rectangle(img_copy, 
                         (left, top), 
                         (left + width, top + height),
                         (0, 255, 0), 2)
            cv2.putText(img_copy, region_name, (left, top - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Reset points
            start_point = None
            end_point = None
    
    cv2.imshow('Region Selector', img_copy)

def generate_code():
    """Generate Python code for the selected regions."""
    if not regions:
        return "No regions selected."
    
    code = "# Manual region definitions\n"
    code += "MANUAL_GAME_REGION = {\n"
    
    # Use 'game_window' or first region as main region
    main_region = regions.get('game_window', next(iter(regions.values())))
    code += f"    \"top\": {main_region['top']},\n"
    code += f"    \"left\": {main_region['left']},\n"
    code += f"    \"width\": {main_region['width']},\n"
    code += f"    \"height\": {main_region['height']}\n"
    code += "}\n\n"
    
    # Generate sub-regions code
    code += "sub_regions = {\n"
    for name, coords in regions.items():
        if name != 'game_window':
            code += f"    \"{name}\": {{\n"
            code += f"        \"top\": {coords['top']},\n"
            code += f"        \"left\": {coords['left']},\n"
            code += f"        \"width\": {coords['width']},\n"
            code += f"        \"height\": {coords['height']}\n"
            code += "    },\n"
    code += "}\n"
    
    return code

def main():
    """Main function to run the region selector tool."""
    global image
    
    print("Region Selector Tool for Poker AI")
    print("----------------------------------")
    print("Instructions:")
    print("1. Click and drag to select a region")
    print("2. Enter a name for the region when prompted")
    print("3. Repeat for all regions you need")
    print("4. Press 'q' to quit and generate code")
    print("5. Press 'c' to clear all regions and start over")
    
    # Take screenshot
    image = take_screenshot()
    
    # Create window and set mouse callback
    cv2.namedWindow('Region Selector')
    cv2.setMouseCallback('Region Selector', mouse_callback)
    
    while True:
        # Make a copy of the image to draw on
        img_copy = image.copy()
        
        # Draw existing regions
        for name, coords in regions.items():
            cv2.rectangle(img_copy, 
                         (coords['left'], coords['top']), 
                         (coords['left'] + coords['width'], coords['top'] + coords['height']),
                         (0, 255, 0), 2)
            cv2.putText(img_copy, name, (coords['left'], coords['top'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Region Selector', img_copy)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Quit on 'q' key
        if key == ord('q'):
            break
        
        # Clear all regions on 'c' key
        if key == ord('c'):
            regions.clear()
            print("All regions cleared")
    
    # Generate and display code
    code = generate_code()
    print("\n----- Generated Python Code -----")
    print(code)
    
    # Save code to file
    with open("poker_regions.py", "w") as f:
        f.write(code)
    print("\nCode saved to poker_regions.py")
    
    # Ask if user wants to update main_manual.py
    root = tk.Tk()
    root.withdraw()
    update_main = messagebox.askyesno("Update main_manual.py", 
                                     "Do you want to update main_manual.py with these regions?")
    root.destroy()
    
    if update_main:
        try:
            # This implementation assumes main_manual.py will import poker_regions.py
            print("main_manual.py will use regions from poker_regions.py")
            print("Run main_manual.py to apply the changes")
        except Exception as e:
            print(f"Error updating main_manual.py: {e}")
    
    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()