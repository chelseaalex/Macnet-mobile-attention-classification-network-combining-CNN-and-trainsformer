from PIL import Image
import openslide
import numpy as np
import pandas
import imageio


target_path="D:/cancer/cancer_cut_test2_256/"
slide = openslide.open_slide("E:/cancer/cut_target/35/20211224_144733_0#13_2.svs")

[m,n]=slide.dimensions
print(m,n)

x=256
ml=x*(m//x)
nl=x*(n//x)
print(ml,nl)
for i in range(0,ml,x):
    for j in range(0,nl,x):
        im=np.array(slide.read_region((i,j),0,(x,x)))
        imageio.imwrite(target_path+str(i)+'-'+str(j)+'.png',im)

slide.close


"""
def convert_color(from_image,classes):
    #from_image=Image.fromarray(from_image.numpy())
    image=from_image.copy()
    r,g,b=image.split()
    if(classes==1):#high
        m=np.asarray(r)
        n=m.copy()
        n[n<255]=255
        r=Image.fromarray(n)
        m=np.asarray(g)
        n=m.copy()
        n[n<256]=0
        g=Image.fromarray(n)
        m=np.asarray(b)
        n=m.copy()
        n[n<256]=0
        b=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    if(classes==3):#mid
        m=np.asarray(r)
        n=m.copy()
        n[n<256]=0
        r=Image.fromarray(n)
        m=np.asarray(g)
        n=m.copy()
        n[n<255]=255
        g=Image.fromarray(n)
        m=np.asarray(b)
        n=m.copy()
        n[n<256]=0
        b=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    if(classes==2):#low
        m=np.asarray(r)
        n=m.copy()
        n[n<256]=0
        r=Image.fromarray(n)
        m=np.asarray(g)
        n=m.copy()
        n[n<256]=0
        g=Image.fromarray(n)
        m=np.asarray(b)
        n=m.copy()
        n[n<255]=255
        b=Image.fromarray(n)
        image = Image.merge('RGB', (r, g, b))
    else:
        image=from_image
    return image
"""
