import mrcfile
import numpy as np
from pylab import *
from scipy import optimize, special
import matplotlib.pyplot as plt
import math
import cv2
import matplotlib.image as mpimg
from matplotlib.widgets import Slider
from matplotlib.widgets import RectangleSelector
from PIL import Image
def get_bessel_order(n):
     bessel_order={}
     x = arange(0,n+2,0.01)    
     for i in arange(n+1):
          y = special.jv(i,x)
          f = lambda x: -special.jv(i,x)
          x_max = optimize.fminbound(f,0,i+2)
          bessel_order[i]=x_max
     return bessel_order
def transform(path):
    try:
        with mrcfile.open(path) as mrc:
            img=mrc.data
    except:
        img=cv2.imread(path,0)
    return img
def readimg(path,option):
    try:
        with mrcfile.open(path) as mrc:
            if option == "Yes":
                img=mrc.data
                f=np.fft.fft2(img)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log1p(np.abs(fshift))
            else:
                magnitude_spectrum=20 * np.log1p(mrc.data)
    except:
        img = cv2.imread(path, 0)
        if option == "Yes":
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log1p(np.abs(fshift))
        else:
            magnitude_spectrum=20*np.log1p(img)
    return magnitude_spectrum
def get_center(path):
    try:
        with mrcfile.open(path) as mrc:
            img=mrc.data
            b,a=img.shape
            center_x,center_y=int(a/2),int(b/2)
    except:
        img = cv2.imread(path, 0)
        b, a = img.shape
        center_x, center_y = int(a / 2), int(b / 2)
    return center_x,center_y
def run(img,center_x,center_y,magnitude_spectrum,nums,r,pixel_num,pixel_size,filename):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Diffraction pattern")
    plt.title('Diffraction pattern'), plt.xticks([]), plt.yticks([])
    plt.subplots_adjust(bottom=0.25)
    im = ax.imshow(magnitude_spectrum,cmap="gray")
    ax_zoom = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_contrast = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_brightness = plt.axes([0.25, 0.05, 0.65, 0.03])

    s_zoom = Slider(ax_zoom, 'Zoom', 1.0, 10.0, valinit=1.0)
    s_contrast = Slider(ax_contrast, 'Contrast', 0.1, 30.0, valinit=1.0)
    s_brightness = Slider(ax_brightness, 'Brightness', -1000, 1000, valinit=0)
    def update(val):
        zoom = s_zoom.val
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
    def onselect(eclick, erelease):
        log = open("./Basic_vector_coordinate/"+"bessel_order"+filename, "a")
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("X_range:" f"({x1:.2f}, {x2:.2f})","Y_range:"f"({y1:.2f}, {y2:.2f})")
        print("X_range:" f"({x1:.2f}, {x2:.2f})","Y_range:"f"({y1:.2f}, {y2:.2f})",file=log)
        global X1,X2
        X1,X2,Y1,Y2=int(x1),int(x2),int(y1),int(y2)
        roi=np.array(magnitude_spectrum[Y1:Y2, X1:X2])
        roi_2=roi.sum(axis=0)
        X=np.arange(X1-center_x,X2-center_x)
        fig2=plt.figure(2)
        ax2 = fig2.add_subplot(111)
        plt.title('Layer Data')
        ax2.plot(X,roi_2.flatten(),label="Experimental data")
        plt.legend(loc='upper right')
        def onclick2(event):
            if event.inaxes == ax2:
                if event.xdata and event.ydata:
                    x1, y1 = event.xdata,event.ydata
                    print(f'You clicked on coordinates ({x1}, {y1})',flush=True)
                    print(f'You clicked on coordinates ({x1}, {y1})',flush=True,file=log)
                    if x1>0:
                        N = (2 * math.pi * x1 * r) / (pixel_num * pixel_size)
                        n_max = (N - 1) / 1.03
                    else:
                        N = (2 * math.pi * abs(x1) * r) / (pixel_num * pixel_size)
                        n_max = -1*((N - 1) / 1.03)
                    my_dict =get_bessel_order(abs(int(n_max)))
                    diff_list = [(key, abs(value - N)) for key, value in my_dict.items ()]
                    print(my_dict)
                    print(diff_list)
                    sorted_list = sorted (diff_list, key=lambda x: x [1])
                    result = [x [0] for x in sorted_list [:4]]
                    print (result)
                    fig3=plt.figure(3)
                    ax3 = fig3.add_subplot(111)
                    plt.title('Bessel Data')
                    R1=(2 * math.pi * (X1-center_x) * r) / (pixel_num * pixel_size)
                    R2=(2 * math.pi * (X2-center_x) * r) / (pixel_num * pixel_size)
                    R=np.arange(int(R1),int(R2),0.01)
                    for i in result:
                        J_label="J"+str(i)
                        x=np.arange(X1-center_x,X2-center_x,0.01)
                        y=special.jv(i,R)
                        # f = lambda x: -special.jv(i,x)
                        # x_max = optimize.fminbound(f,0,i+4)
                        ax3.plot(R,y,label=J_label)
                        plt.legend(loc='upper right')
                    show()   
        fig2.canvas.mpl_connect('button_press_event', onclick2) 
        show()
    rect_selector = RectangleSelector(
          ax,
          onselect,
          interactive=True,
          props=dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)
          )
    def onclick(event):
        if event.inaxes == ax:
            if event.xdata and event.ydata:
                if event.dblclick:
                    x1, y1 = int(event.xdata),int(event.ydata)
                    X, Y = int(event.xdata)-center_x, center_y-int(event.ydata)
                    infor=[X,Y]+nums
                    print(f'You clicked on coordinates ({x1}, {y1})',flush=True)
                    print(f'dcenter coordinates ({X}, {Y})',flush=True)
                    Z = Y / (pixel_num * pixel_size)
                    if X>0:
                        N = (2 * math.pi * X * r) / (pixel_num * pixel_size)
                        n_max = (N - 1) / 1.03
                    else:
                        N = (2 * math.pi * abs(X) * r) / (pixel_num * pixel_size)
                        n_max = -1*((N - 1) / 1.03)
                    results = [n_max, Z]
                    a = open("./Basic_vector_coordinate/"+filename, "a")
                    print(f'You clicked on coordinates ({x1}, {y1})',file=a,flush=True)
                    print(f'dcenter coordinates ({X}, {Y})',file=a,flush=True)
                    print("X,Y,pixel number,pixel size,spiral radius (unit: angstroms)", "\n", infor, "\n", "n Z",
                        results, file=a,flush=True)
                    a.close()

                    print("X,Y,pixel number,pixel size,spiral radius (unit: angstroms)", "\n", infor, "\n", "n Z",
                        results,flush=True)

    fig.canvas.mpl_connect("button_press_event", onclick)
    def on_key(event):
        if event.key == ' ':
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_min = int(np.around(x_min))
            x_max = int(np.ceil(x_max))
            y_min = int(np.around(y_min))
            y_max = int(np.ceil(y_max))
            img_data = (magnitude_spectrum - magnitude_spectrum.mean()) * s_contrast.val + magnitude_spectrum.mean()+s_brightness.val
            # if img_data.dtype == np.float32 or img_data.dtype == np.float64:
            #     if img_data.max()<255:
            #         img_data=(img_data*255).astype(np.float64)
            #     else:
            #         pass
            data=img_data[y_max:y_min, x_min:x_max]
            
            data_min=np.min(data)
            data_max=np.max(data)
            
            norm_data=(data-data_min)/(data_max-data_min)
            
            save_img_data=norm_data*255
            
            image = Image.fromarray(save_img_data)
            image = image.convert('RGB')
            image.save("./Basic_vector_coordinate/diffraction_pattern.png")
            with open('./Basic_vector_coordinate/values.txt', 'a') as f:
                print('Image and values saved.')
                print('Zoom:', s_zoom.val,file=f,flush=True)
                print('Contrast:', s_contrast.val,file=f,flush=True)
                print('Brightness:', s_brightness.val,file=f,flush=True)
                print("region:","{0}, {1}, {2}, {3}".format(y_max,y_min,x_min,x_max),file=f,flush=True)
            f.close()
    plt.connect('key_press_event', on_key)




    plt.show()
