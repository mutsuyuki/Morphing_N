from PIL import Image, ImageDraw, ImageFont

f = open('paths.txt')

imageSize =512 

counter = 0
line = f.readline()

while line:

    try:
        path = line
        path = path.replace('\n', '')
        path = path.replace('\r', '')
        path = path.replace('*', '')
        font = ImageFont.truetype(path, imageSize - 10)

        image = Image.new('RGBA', (imageSize,imageSize), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        textWidth, textHeight = font.getsize("N")
        textPosition = ((imageSize - textWidth) / 2, (imageSize - textHeight) / 2 - 25)

        draw.text(textPosition, 'N', font=font, fill='#000000')

        image.save("my_fonts/" + str(counter) + '.png', 'PNG')
    except:
        print("error at", line)

    counter += 1

    line = f.readline()

f.close()
