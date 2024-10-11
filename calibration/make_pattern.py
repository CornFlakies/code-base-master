# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:04:03 2024

@author: coena
"""

import os
from gen_pattern import PatternMaker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

path ="C:\\Users\\coena\\WORK\\master\\code-base-master\\patterns\\"
# filename = "checkerboard_pattern.svg"
filename = "circles_pattern.svg"

units               = 'mm' # defaults to mm, take care with setting the sizes
output              = os.path.join(path, filename)
columns             = 50
rows                = 50
p_type              = "circles"
square_size         = 1
radius_rate         = square_size / 4  
page_size           = [594, 840] # a4
page_width          = page_size[0]
page_height         = page_size[1]
markers             = None
aruco_marker_size   = None
dict_file           = None
args                = [None]

print('column length: ' + str(columns * square_size) + ' mm')
print('row length: ' + str(rows * square_size) + ' mm')

pm = PatternMaker(columns, rows, output, units, square_size, radius_rate, page_width, page_height, markers, aruco_marker_size, dict_file)

# dict for easy lookup of pattern type
mp = {"circles": pm.make_circles_pattern, "acircles": pm.make_acircles_pattern,
      "checkerboard": pm.make_checkerboard_pattern, "radon_checkerboard": pm.make_radon_checkerboard_pattern,
     "charuco_board": pm.make_charuco_board}
mp[p_type]()

# this should save pattern to output
pm.save()

#%%
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4

# List of SVG files to combine
svg_files=['checkerboard_pattern_1000microns',
           'checkerboard_pattern_500microns',
           'checkerboard_pattern_250microns',
           'circles_pattern_1000microns',
           'circles_pattern_500microns',
           'circles_pattern_250microns']

# Output PDF file name
output_pdf = os.path.join(path, 'output_grid.pdf')

# Create a PDF canvas with A4 page size
c = canvas.Canvas(output_pdf, pagesize=A4)

# Define page dimensions (A4 size in points)
page_width, page_height = A4

# Define number of columns and rows
columns = 2
rows = 3

# Calculate the width and height for each SVG drawing based on the number of columns and rows
svg_width = page_width / columns
svg_height = page_height / rows

# Define the spacing between SVGs (optional, if needed)
x_spacing = 25
y_spacing = 25

# Loop over the SVG files and arrange them in a grid
for idx, svg_file in enumerate(svg_files):
    # Convert SVG to a ReportLab drawing
    svg_file = os.path.join(path, svg_file + '.svg') 
    drawing = svg2rlg(svg_file)

    # Calculate the row and column position for this SVG
    col = idx % columns
    row = idx // columns

    # Calculate X and Y positions
    x_position = col * svg_width + x_spacing
    y_position = page_height - ((row + 1) * svg_height - y_spacing)  # Invert Y axis for PDF

    # # Scale the drawing to fit in the calculated grid cell
    # scale_x = svg_width / drawing.width
    # scale_y = svg_height / drawing.height
    # scale_factor = min(scale_x, scale_y)  # Ensure proportional scaling

    # # Apply scaling
    # drawing.width *= scale_factor
    # drawing.height *= scale_factor
    # drawing.scale(scale_factor, scale_factor)

    # Draw the SVG drawing at the calculated position
    print(x_position)
    print(y_position)
    renderPDF.draw(drawing, c, 20, 20)

# Save the PDF file
c.save()

print(f"Successfully arranged SVG files into a 2x3 grid on {output_pdf}.")

# # Define the input SVG file and the output PDF file
# for ii in range(6):
#     ii = 3
    # input_svg = os.path.join(path, svg_files[ii] + '.svg')  # replace with your SVG file name
#     output_pdf = os.path.join(path, svg_files[ii] + '.pdf')  # replace with your desired PDF file name

#     # Convert SVG to a ReportLab drawing
#     drawing = svg2rlg(input_svg)

#     # Render the drawing to a PDF file
#     renderPDF.drawToFile(drawing, output_pdf)

#     print(f"Successfully converted {input_svg} to {output_pdf}.")
    
#     break
