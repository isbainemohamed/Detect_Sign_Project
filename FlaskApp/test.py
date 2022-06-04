from PIL import Image
import numpy as np
WIDTH=32
HEIGHT=32
# Image.open() can also open other image types
img = Image.open("uploads/test_docs/img.png")
im=np.array(img)
print("initial size is ,", im.shape)
# WIDTH and HEIGHT are integers
resized_img = img.resize((WIDTH, HEIGHT))
res_im=np.array(resized_img)
resized_img.save("uploads/test_docs/resized_image1.png")
print("initial size is ,", res_im.shape)
rgb_image = resized_img.convert('RGB')
print("rgb  size is ,", np.array(rgb_image).shape)
