# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:56:22 2024

@author: coena
"""

from svglib.svglib import svg2rlg
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

# Function to combine SVG images into a PDF
def combine_svgs_to_pdf(svg_files, output_pdf):
    # Create a canvas for the PDF with A4 size
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4

    # Iterate through the SVG files and add them to the PDF
    for svg_file in svg_files:
        # Convert SVG to ReportLab drawing
        drawing = svg2rlg(svg_file)
        
        # Scale the drawing to fit within A4 dimensions
        scale = min(width / drawing.width, height / drawing.height)
        drawing.width *= scale
        drawing.height *= scale
        
        # Position the drawing in the center of the page
        drawing.x = (width - drawing.width) / 2
        drawing.y = (height - drawing.height) / 2
        
        # Draw the SVG onto the canvas
        renderPDF.draw(drawing, c, 0, 0)
        
        # Create a new page for the next SVG
        c.showPage()

    # Save the PDF file
    c.save()

# List of SVG files to combine
svg_files = ['image1.svg', 'image2.svg', 'image3.svg']  # Update with your SVG file paths
output_pdf = 'combined_images.pdf'

# Combine the SVGs and create the PDF
combine_svgs_to_pdf(svg_files, output_pdf)
print(f"Combined SVG images into {output_pdf}")
