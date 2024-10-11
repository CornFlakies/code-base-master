import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.close('all')

# Jo
path="C:\\Users\\coena\\WORK\\master\\code-base-master\\patterns\\pattern_a4_dual_20x20mm_with_spacing.pdf"

# A4 dimensions in millimeters
A4_width_mm = 210
A4_height_mm = 297

# Function to create the dual pattern with margins on A4 paper
def create_pattern_on_a4():
    fig, ax = plt.subplots(figsize=(A4_width_mm/25.4, A4_height_mm/25.4))  # Convert mm to inches for matplotlib

    # Set the limits to match A4 size
    ax.set_xlim(0, A4_width_mm)
    ax.set_ylim(0, A4_height_mm)
    
    # Set equal scaling
    ax.set_aspect('equal')

    # Disable axis
    ax.axis('off')

    # Define margins in mm
    margin = 25  # 2.5 cm margin from top and left

    # Define the top-left corner for both patterns
    pattern_top_left_x_1 = margin
    pattern_top_left_y = A4_height_mm - 20 - margin  # Position the pattern with margin

    # Second pattern is 50 mm (5 cm) to the right of the first one
    pattern_top_left_x_2 = pattern_top_left_x_1 + 20 + 50  # 20 mm for the first pattern + 50 mm spacing

    # Third pattern is 50 mm (5 cm) to the bottom of the first one
    pattern_top_left_y_2 = pattern_top_left_y - 20 - 50  # Position the pattern with margin

    # Pattern 1: 0.25 mm squares, covering 20x20 mm
    square_size_1 = 0.20
    for i in range(100):  # 20 mm / 0.25 mm = 80 squares along x
        for j in range(100):  # 20 mm / 0.25 mm = 80 squares along y
            square = patches.Rectangle(
                (pattern_top_left_x_1 + i * square_size_1, pattern_top_left_y + j * square_size_1),
                square_size_1, square_size_1,
                edgecolor='black', facecolor='none'
            )
            ax.add_patch(square)

    # Pattern 2: 0.5 mm squares, covering 20x20 mm
    square_size_2 = 0.5
    for i in range(40):  # 20 mm / 0.5 mm = 40 squares along x
        for j in range(40):  # 20 mm / 0.5 mm = 40 squares along y
            square = patches.Rectangle(
                (pattern_top_left_x_2 + i * square_size_2, pattern_top_left_y + j * square_size_2),
                square_size_2, square_size_2,
                edgecolor='black', facecolor='none'
            )
            ax.add_patch(square)
                     
    # Circle radius and spacing
    circle_diameter = 0.1  # Radius in mm
    circle_spacing = 0.2  # 2 mm spacing between circles

    # Pattern 1: Circles covering 20x20 mm
    for i in range(100):  # 20 mm / 2 mm = 10 circles along x
        for j in range(100):  # 20 mm / 2 mm = 10 circles along y
            # Calculate position of each circle
            circle_x = pattern_top_left_x_1 + i * circle_spacing
            circle_y = pattern_top_left_y_2 + j * circle_spacing
            
            # Create the circle
            circle = patches.Circle((circle_x, circle_y), radius=circle_diameter/2, edgecolor='black', facecolor='black')
            ax.add_patch(circle)
            
    # Circle radius and spacing
    circle_diameter = 0.1  # Radius in mm
    circle_spacing = 0.5  # 2 mm spacing between circles

    # Pattern 1: Dots covering 20x20 mm
    for i in range(40):  # 20 mm / 2 mm = 10 dots along x
        for j in range(40):  # 20 mm / 2 mm = 10 dots along y
            # Calculate position of each dot
            circle_x = pattern_top_left_x_2 + i * circle_spacing
            circle_y = pattern_top_left_y_2 + j * circle_spacing
            
            # Create the circle
            circle = patches.Circle((circle_x, circle_y), radius=circle_diameter/2, edgecolor='black', facecolor='black')
            ax.add_patch(circle)

    # Save the pattern as a PDF to preserve the scale for printing
    plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0)
    
    # Display the pattern (optional for visualization before printing)
    plt.show()

# Call the function to create and display the pattern
create_pattern_on_a4()
