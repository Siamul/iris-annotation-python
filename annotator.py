from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import os
from PIL import ImageTk,Image,ImageDraw, ImageOps
import numpy as np
from scipy import optimize
import math
import json
import cv2

root = Tk()

#top_label = Label(root, text="Iris Annotation Tool")

#top_label.pack()

root.title("Iris Annotation Tool")

images = []
current_img = None
image_index = 0
folder_path = ''

points = {}  #index 0 is pupil and 1 is iris
points['pupil'] = []
points['iris'] = []
points_index = 'pupil'
delete_bool=False
delete_dist_limit = 42
current_img = None
image_name_only = []
clahe_bool = False
image_mult = 1

p_x, p_y, p_r, i_x, i_y, i_r = 0, 0, 0, 0, 0, 0

def get_euclid_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x, y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def create_image_with_points():
    global images, image_index, points, p_x, p_y, p_r, i_x, i_y, i_r, clahe_bool, image_mult
    img = Image.open(images[image_index]).convert('RGB')
    if clahe_bool:
        print("CLAHE!")
        img_gray = img.convert('L')
        #img = ImageOps.autocontrast(img_gray).convert('RGB')
        
        img_np = np.asarray(img_gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)
        img = Image.fromarray(img_np).convert('RGB')

    width, height = img.size
    width = width * image_mult
    height = height * image_mult
    img = img.resize((width, height))    
    draw = ImageDraw.Draw(img)

    for point in points['pupil']:
        point = [image_mult * coordv for coordv in point]
        draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill='red')

    for point in points['iris']:
        point = [image_mult * coordv for coordv in point]
        draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill='blue')

    if len(points['pupil']) >= 3:
        pupil_x_points = []
        pupil_y_points = []
        for point in points['pupil']:
            pupil_x_points.append(point[0])
            pupil_y_points.append(point[1])
        p_x, p_y, p_r, p_res = [retval * image_mult for retval in leastsq_circle(pupil_x_points, pupil_y_points)]

        draw.ellipse((p_x-p_r, p_y-p_r, p_x+p_r, p_y+p_r), outline='red')
        
    if len(points['iris']) >= 3:
        iris_x_points = []
        iris_y_points = []
        for point in points['iris']:
            iris_x_points.append(point[0])
            iris_y_points.append(point[1])
        i_x, i_y, i_r, i_res = [retval * image_mult for retval in leastsq_circle(iris_x_points, iris_y_points)]
        draw.ellipse((i_x-i_r, i_y-i_r, i_x+i_r, i_y+i_r), outline='blue')

    return ImageTk.PhotoImage(img)

def select_image_folder():
    global images, image_index, previous_image_button, next_image_button, pupil_button, iris_button, \
                    current_img, canvas, delete_button, image_name_only, folder_path, points
    if len(images) != 0:    
        data_file_prev = os.path.join(folder_path, image_name_only[image_index].split('.')[0] + '.json')
        data = {}
        data['points'] = points
        data['pupil_xyr'] = [p_x, p_y, p_r]
        data['iris_xyr'] = [i_x, i_y, i_r]
        json.dump( data, open( data_file_prev, 'w' ) )
    images = []
    image_index = 0
    filename = filedialog.askdirectory()
    folder_path = filename
    for imagename in os.listdir(folder_path):
        if imagename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            images.append(os.path.join(folder_path, imagename))
            image_name_only.append(imagename)
    Label(root, text=images[image_index]).grid(row=1, columnspan=7)
    points['pupil'] = []
    points['iris'] = []
    data_file = os.path.join(folder_path, image_name_only[image_index].split('.')[0] + '.json')
    
    if os.path.exists(data_file):
        data = json.load(open(data_file))
        points = data['points']
    current_img = create_image_with_points()
    canvas.config(width = current_img.width(), height = current_img.height())
    canvas.create_image(0, 0, anchor=NW, image=current_img)
    previous_image_button['state']='normal'
    next_image_button['state']='normal'
    pupil_button['state']='normal'
    iris_button['state']='normal'
    delete_button['state']='normal'
    clear_button['state']='normal'
    enlarge_button['state']='normal'
    reduce_button['state']='normal'
    clahe_button['state']='normal'
    
    #canvas['state']='normal'

def previous_image():
    global images, image_index, current_img, canvas, points
    if image_index > 0:
        data_file_prev = os.path.join(folder_path, image_name_only[image_index].split('.')[0] + '.json')
        data = {}
        data['points'] = points
        data['pupil_xyr'] = [p_x, p_y, p_r]
        data['iris_xyr'] = [i_x, i_y, i_r]
        json.dump( data, open( data_file_prev, 'w' ) )
        image_index -= 1
        Label(root, text=images[image_index]).grid(row=1, columnspan=7)
        points['pupil'] = []
        points['iris'] = []
        data_file = os.path.join(folder_path, image_name_only[image_index].split('.')[0] + '.json')
        if os.path.exists(data_file):
            data = json.load(open(data_file))
            points = data['points']
        
        current_img = create_image_with_points()
        canvas.config(width = current_img.width(), height = current_img.height())
        canvas.create_image(0, 0, anchor=NW, image=current_img)

def next_image():
    global images, image_index, current_img, canvas, points, p_x, p_y, p_r, i_x, i_y, i_r
    if image_index < (len(images) - 1):
        data_file_prev = os.path.join(folder_path, image_name_only[image_index].split('.')[0] + '.json')
        data = {}
        data['points'] = points
        data['pupil_xyr'] = [p_x, p_y, p_r]
        data['iris_xyr'] = [i_x, i_y, i_r]
        json.dump( data, open( data_file_prev, 'w' ) )
        image_index += 1
        Label(root, text=images[image_index]).grid(row=1, columnspan=7)
        points['pupil'] = []
        points['iris'] = []
        data_file = os.path.join(folder_path, image_name_only[image_index].split('.')[0] + '.json')
        if os.path.exists(data_file):
            data = json.load(open(data_file))
            points = data['points']
        current_img = create_image_with_points()
        canvas.config(width = current_img.width(), height = current_img.height())
        canvas.create_image(0, 0, anchor=NW, image=current_img)

def select_pupil_list():
    global points_index, pupil_button, iris_button, delete_button, delete_bool
    pupil_button['relief']='sunken'
    iris_button['relief']='raised'
    delete_button['relief']='raised'
    points_index = 'pupil'
    delete_bool=False

def select_iris_list():
    global points_index, pupil_button, iris_button, delete_button, delete_bool
    pupil_button['relief']='raised'
    iris_button['relief']='sunken'
    delete_button['relief']='raised'
    points_index = 'iris'
    delete_bool=False

def delete_point():
    global pupil_button, iris_button, delete_button, delete_bool
    pupil_button['relief']='raised'
    iris_button['relief']='raised'
    delete_button['relief']='sunken'
    delete_bool=True

#def key(event):
    #print("pressed", repr(event.char))

def callback(event):
    #print("clicked at", event.x, event.y)
    global current_img, points, points_index, delete_bool, image_mult
    x = int(event.x / image_mult) 
    y = int(event.y / image_mult)
    if delete_bool==False:
        points[points_index].append([x,y])
    else:
        if len(points['pupil']) > 0 and len(points['iris']) > 0:
            pupil_dist = []
            for point in points['pupil']:
                pupil_dist.append(get_euclid_dist([x,y], point))
            min_euclid_dist_pupil = min(pupil_dist)
            closest_pupil_point_index = pupil_dist.index(min_euclid_dist_pupil)
            
            iris_dist = []
            for point in points['iris']:
                iris_dist.append(get_euclid_dist([x,y], point))
            min_euclid_dist_iris = min(iris_dist)
            closest_iris_point_index = iris_dist.index(min_euclid_dist_iris)

            closest_pupil_dist = pupil_dist[closest_pupil_point_index]
            closest_iris_dist = iris_dist[closest_iris_point_index]
            
            if closest_pupil_dist < delete_dist_limit and closest_iris_dist < delete_dist_limit:
                if closest_pupil_dist < closest_iris_dist:
                    points['pupil'].pop(closest_pupil_point_index)
                else:
                    points['iris'].pop(closest_iris_point_index)
            elif closest_pupil_dist < delete_dist_limit:
                points['pupil'].pop(closest_pupil_point_index)
            elif closest_iris_dist < delete_dist_limit:
                points['iris'].pop(closest_iris_point_index)
        elif len(points['pupil']) > 0:
            pupil_dist = []
            for point in points['pupil']:
                pupil_dist.append(get_euclid_dist([x,y], point))
            min_euclid_dist_pupil = min(pupil_dist)
            closest_pupil_point_index = pupil_dist.index(min_euclid_dist_pupil)
            closest_pupil_dist = pupil_dist[closest_pupil_point_index]
            if closest_pupil_dist < delete_dist_limit:
                points['pupil'].pop(closest_pupil_point_index)
        elif len(points['iris']) > 0:
            iris_dist = []
            for point in points['iris']:
                iris_dist.append(get_euclid_dist([x,y], point))
            min_euclid_dist_iris = min(iris_dist)
            closest_iris_point_index = iris_dist.index(min_euclid_dist_iris)
            closest_iris_dist = iris_dist[closest_iris_point_index]
            if closest_iris_dist < delete_dist_limit:
                points['iris'].pop(closest_iris_point_index)
            
        
    if current_img is not None:
        current_img = create_image_with_points()    
        canvas.create_image(0, 0, anchor=NW, image=current_img)

def clear_all_points():
    global points, current_img
    points['pupil'] = []
    points['iris'] = []
    current_img = create_image_with_points()    
    canvas.create_image(0, 0, anchor=NW, image=current_img)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to save and quit?"):
        try:
            data_file_prev = os.path.join(folder_path, image_name_only[image_index].split('.')[0] + '.json')
            data = {}
            data['points'] = points
            data['pupil_xyr'] = [p_x, p_y, p_r]
            data['iris_xyr'] = [i_x, i_y, i_r]
            json.dump( data, open( data_file_prev, 'w' ) )
        except Exception as e:
            pass
        root.destroy()

def apply_CLAHE():
    global clahe_bool, canvas, current_img
    if clahe_var.get() == 1:
        clahe_bool = True
    else:
        clahe_bool = False
    print("CLAHE bool is now: ", str(clahe_bool))
    current_img = create_image_with_points()
    canvas.config(width = current_img.width(), height = current_img.height())
    canvas.create_image(0, 0, anchor=NW, image=current_img)
        
def enlarge_image():
    global image_mult, canvas, current_img
    if image_mult < 10:
        image_mult += 1
        print("Image Multiplier: ", image_mult)
    current_img = create_image_with_points()
    canvas.config(width = current_img.width(), height = current_img.height())
    canvas.create_image(0, 0, anchor=NW, image=current_img)

def reduce_image():
    global image_mult, canvas, current_img
    if image_mult > 1:
        image_mult -= 1
        print("Image Multiplier: ", image_mult)
    current_img = create_image_with_points()
    canvas.config(width = current_img.width(), height = current_img.height())
    canvas.create_image(0, 0, anchor=NW, image=current_img)


select_image_folder_button = Button(root, text="Select Image Folder", command=select_image_folder)
previous_image_button = Button(root, text="Previous Image", state=DISABLED, command=previous_image)
next_image_button = Button(root, text="Next Image", state=DISABLED, command=next_image)
pupil_button = Button(root, text ="Set pupil circle", state=DISABLED, relief=SUNKEN, command=select_pupil_list)
iris_button = Button(root, text="Set iris circle", state=DISABLED, relief=RAISED, command=select_iris_list)
delete_button = Button(root, text="Delete point", state=DISABLED, relief=RAISED, command=delete_point)
clear_button = Button(root, text="Clear all points", state=DISABLED, command=clear_all_points)
enlarge_button = Button(root, text="Enlarge Image",  state=DISABLED, command=enlarge_image)
reduce_button = Button(root, text="Reduce Image", state=DISABLED, command=reduce_image)
clahe_var = IntVar()
clahe_button = Checkbutton(root, text='CLAHE',variable=clahe_var, state=DISABLED, onvalue=1, offvalue=0, command=apply_CLAHE)
select_image_folder_button.grid(row=0, column=0)
previous_image_button.grid(row=0, column=1)
next_image_button.grid(row=0, column=2)
pupil_button.grid(row=0, column=3)
iris_button.grid(row=0, column=4)
delete_button.grid(row=0, column=5)
clear_button.grid(row=0, column=6)
clahe_button.grid(row=0, column=7)
enlarge_button.grid(row=0, column=8)
reduce_button.grid(row=0, column=9)
canvas = Canvas(root)
canvas.bind("<Button-1>", callback)
canvas.grid(row=2, columnspan=10)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
