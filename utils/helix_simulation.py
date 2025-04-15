import numpy as np
from PIL import Image, ImageDraw
from pylab import *
from scipy import optimize, special
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from matplotlib.widgets import Slider
from matplotlib.widgets import RectangleSelector
import cv2
import mrcfile
def helix_generator(l,real_r,real_P,real_z,N,size,pixel_size,filename):
    img = Image.new('RGB', (l, l), 'black')
    draw = ImageDraw.Draw(img)
    z=real_z/pixel_size
    P=real_P/pixel_size
    r=real_r/pixel_size
    t=np.arange(0,l,z)
    theta=2*np.pi*t/P
    d=2*np.pi/N
    X=r*np.cos(theta)
    Y=r*np.sin(theta)
    x=l/2+r*np.cos(theta)
    y=l/2+r*np.sin(theta)
    if N ==1:
        for j in range(len(t)):
            xz = (x[j] - (size / 2), t[j] - (size / 2), x[j] + (size / 2), t[j] + (size / 2))
            draw.ellipse(xz,fill="white")
        # img.show()
        img.save(filename)
    else:
        xz= {}
        for i in range(int(N)):
            xz[i] =l/2+ (X * np.cos(i * d) - Y * np.sin(i * d))
            # y1 = x * np.sin(i * d) + y * np.cos(i * d)
            for j in range(len(t)):
                coord=(xz[i][j] - size/2, t[j] - size/2, xz[i][j] + size/2, t[j] + size/2)
                draw.ellipse(coord, fill="white")

        # img.show()
        img.save(filename)
def get_img(path):
    try:
        with mrcfile.open(path) as mrc:
            img=mrc.data
    except:
        img=cv2.imread(path,0)
    return img
def FFT(path):
    try:
        with mrcfile.open(path) as mrc:
            img=mrc.data
            f=np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log1p(np.abs(fshift))

    except:
        img = cv2.imread(path, 0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log1p(np.abs(fshift))

    else:
        pass

    return magnitude_spectrum

def run(img,magnitude_spectrum):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Diffraction pattern")
    plt.title('Diffraction pattern'), plt.xticks([]), plt.yticks([])
    plt.subplots_adjust(bottom=0.25)
    im = ax.imshow(magnitude_spectrum,cmap="gray")
    ax_zoom = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_contrast = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_brightness = plt.axes([0.25, 0.05, 0.65, 0.03])

    s_zoom = Slider(ax_zoom, 'Zoom', 1.0, 10.0, valinit=1.0)
    s_contrast = Slider(ax_contrast, 'Contrast', 0.1, 20.0, valinit=1.0)
    s_brightness = Slider(ax_brightness, 'Brightness', -1000, 1000, valinit=0)
    def update(val):
        zoom = s_zoom.val # 获取滑条的值
        contrast = s_contrast.val
        brightness = s_brightness.val
        xcenter, ycenter = ax.get_xlim(), ax.get_ylim()
        xcenter = (xcenter[0] + xcenter[1]) / 2
        ycenter = (ycenter[0] + ycenter[1]) / 2
        xwidth = img.shape[1] / zoom
        yheight = img.shape[0] / zoom
        xmin = max(0, xcenter - xwidth / 2)
        xmax = min(img.shape[1], xcenter + xwidth / 2)
        ymin = max(0, ycenter - yheight / 2)
        ymax = min(img.shape[0], ycenter + yheight / 2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        im.set_data((magnitude_spectrum - magnitude_spectrum.mean()) * contrast + magnitude_spectrum.mean()+brightness)
        fig.canvas.draw_idle()
    def onscroll(event):
        zoom = s_zoom.val
        if event.button == 'up':
            zoom += 0.1
        elif event.button == 'down':
            zoom -= 0.1
        s_zoom.set_val(zoom)

    fig.canvas.mpl_connect('scroll_event', onscroll)
    s_zoom.on_changed(update)
    s_contrast.on_changed(update)
    s_brightness.on_changed(update)


    def on_key(event):
        if event.key == ' ':
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_min = int(np.around(x_min))
            x_max = int(np.ceil(x_max))
            y_min = int(np.around(y_min))
            y_max = int(np.ceil(y_max))
            img_data = (magnitude_spectrum - magnitude_spectrum.mean()) * s_contrast.val + magnitude_spectrum.mean()+s_brightness.val

            if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                if img_data.max()<255:
                    img_data=(img_data*255).astype(np.float64)
                else:
                    pass
            image = Image.fromarray(img_data[y_max:y_min, x_min:x_max])
            image = image.convert('RGB')
            image.save("./Diffraction_simulation/test_sim.png")
            with open('./Diffraction_simulation/values.txt', 'a') as f:
                print('Image and values saved.')
                print('Zoom:', s_zoom.val,file=f,flush=True)
                print('Contrast:', s_contrast.val,file=f,flush=True)
                print('Brightness:', s_brightness.val,file=f,flush=True)
                print("region:","{0}, {1}, {2}, {3}".format(y_max,y_min,x_min,x_max),file=f,flush=True)
            f.close()

    plt.connect('key_press_event', on_key)

    plt.show()
def batch(img,magnitude_spectrum,batch_zoom,batch_contrast,batch_brightness,region,filename):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Diffraction pattern")
    plt.title('Diffraction pattern'), plt.xticks([]), plt.yticks([])
    plt.subplots_adjust(bottom=0.25)
    contrast = batch_contrast
    brightness = batch_brightness
    zoom=batch_zoom
    xcenter, ycenter = ax.get_xlim(), ax.get_ylim()  #
    xcenter = (xcenter[0] + xcenter[1]) / 2
    ycenter = (ycenter[0] + ycenter[1]) / 2
    xwidth = img.shape[1] / zoom
    yheight = img.shape[0] / zoom
    xmin = max(0, xcenter - xwidth / 2)
    xmax = min(img.shape[1], xcenter + xwidth / 2)
    ymin = max(0, ycenter - yheight / 2)
    ymax = min(img.shape[0], ycenter + yheight / 2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    img_data = (magnitude_spectrum - magnitude_spectrum.mean()) * contrast + magnitude_spectrum.mean() + brightness
    im = ax.imshow(magnitude_spectrum, cmap="gray")
    im.set_data(img_data)
    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
        if img_data.max() < 255:
            img_data = (img_data * 255).astype(np.float64)
        else:
            pass
    image = Image.fromarray(img_data[region[0]:region[1],region[2]:region[3]])
    # image = Image.fromarray(img_data[y_max:y_min, x_min:x_max])
    image = image.convert('RGB')
    try:
        image.save("./Diffraction_simulation/" + filename)
    except:
        name=filename.split(".")[0]+".png"
        image.save("./Diffraction_simulation/" + name)


    plt.close()