import sys
from PIL import Image


f_path = sys.argv[1]

with Image.open(f_path) as img:
	width, height = img.size
	PIX = img.load()
	new_img = Image.new("RGB", (width, height))

	for x in range(0, width):
		for y in range(0, height):
			RGB = (int(PIX[x,y][0]/2), int(PIX[x,y][1]/2), int(PIX[x,y][2]/2) )
			new_img.putpixel([x, y], RGB)

	new_img.save("Q2.png")

