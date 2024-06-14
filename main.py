import pandas as pd
import numpy as np
import random
from   glob import glob
import cv2
import matplotlib.pylab as plt
import os


GROUND_PATH = os.path.dirname(__file__) + "/"
global color_at_mouse, fig, axs, using_colors

class MyColor:
    def __init__(self, rgb: tuple, name: str):
        self.rgb = rgb
        self.luma = calc_luma(rgb)
        self.name = name


class ColorManager:
    def __init__(self):
        self.colors = []
    
    def add(self, color):
        self.colors.append(color)
    
    def remove(self, color):
        self.colors.remove(color)
    
    def add_from_names_list(self, chosen_colors, names_list):
        for name in names_list:
            for color in self.colors:
                if color.name == name:
                    chosen_colors.add(color)
                    break
    
    def print(self, img_name: str):
        color_string = " ".join(color.name for color in self.colors)
        amount_colors = len(self.colors)  
        print("-->",img_name," has ",amount_colors," colors: ",color_string)   
    
    def modify_randomly(self, all_colors, num_changes):        
        available_colors = [color for color in all_colors.colors if color not in self.colors]        
        for _ in range(num_changes):
            if not available_colors:
                break
            
            choice = random.choice(["add", "remove"])
            if choice == "add":
                # Füge eine zufällige Farbe aus all_colors hinzu, die noch nicht in using_colors ist
                random_color = random.choice(available_colors)
                self.add(random_color)
                available_colors.remove(random_color)
                print(f"--> Added the random color: {random_color.name}")
            elif choice == "remove":
                # Entferne eine zufällige Farbe aus using_colors
                if self.colors:
                    random_color = random.choice(self.colors)
                    self.remove(random_color)
                    print(f"--> Removed the random color: {random_color.name}")
                
    def create_copy(self):
        copy = ColorManager()  # Erstelle eine neue Instanz
        for color in self.colors:
            copy.add(color)  #Füge die Farben aus der Übergebene Instanz
        return copy    

def calc_luma(rgb_tuple)->int:#function that calculates the brightness of a pixel as an integer
    r, g, b = rgb_tuple
    luma = int(r * 0.2126 + g * 0.7152 + b * 0.0722)
    return luma    

def luma2rgb(luma,using_colors)->tuple:#converts luma of a pixel into its new color out of chosen colors 
    for color in using_colors.colors:
        if luma < color.luma:
            return color.rgb
    return using_colors.colors[-1].rgb        

def new_img(img_orginal: np.ndarray, using_colors)->np.ndarray:#makes art img and gives new color to it
    img_art = img_orginal.copy()
    for i in range(img_orginal.shape[0]):
        for j in range(img_orginal.shape[1]):
            pixel_color = tuple(img_orginal[i, j])#returns pixel in RGB
            luma = calc_luma(pixel_color)
            new_color = luma2rgb(luma,using_colors)
            img_art[i, j] = new_color
    return img_art

def plot_images(img1: np.ndarray, img2: np.ndarray)-> None:#plots orginal and art image in one plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    axs[0].imshow(img1)
    axs[0].set_title("Original Image")
    axs[1].imshow(img2)
    axs[1].set_title("Abstract Image")
    plt.suptitle(title_maker(), fontsize=30,y=1)
    plt.show(block= False)#Mit diesem Argument läuft das Programm weiter.
    return axs, fig

def update_art_img(img_art):
    axs[1].clear()
    axs[1].imshow(img_art)
    plt.suptitle(title_maker(), fontsize=30,y=1)
    plt.show(block  = False)

def save_img(amount_colors: int,rating: int,luma_mean:int)-> None:# Save the combined figure as a JPG image
    filename = f"R{rating}_C{amount_colors}_L{luma_mean}.jpg"
    filename = filename.replace(" ", "_")
    output_path = GROUND_PATH + "Output_Images/" + filename
    plt.savefig(output_path, format='jpg')
    plt.close()
    print("--> Saved the shown picture in Art_Images Folder Filename is: " + filename)

def resize_image(img: np.ndarray, max_size:int):#resizes the img to max pixels
    height, width = img.shape[:2]    
    if height > width:
        new_height = max_size
        ratio = new_height / height
        new_width = int(width * ratio)
    else:
        new_width = max_size
        ratio = new_width / width
        new_height = int(height * ratio)
    
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def get_random_img()->np.ndarray:#gets random .jpg img from directory
    amount_source_files = len(glob(os.path.join(GROUND_PATH + "Input_Images/*.jpg")))
    selected_index = int(random.uniform(0,amount_source_files))
    img_orginal_bgr = cv2.imread(glob(GROUND_PATH + "Input_Images/*.jpg")[selected_index])
    img_orginal = cv2.cvtColor(img_orginal_bgr, cv2.COLOR_BGR2RGB)#Put img into cv2 format and turn bgr to rgb
    img_orginal = resize_image(img_orginal, 500)#resize image to max size to keep programm faster
    return img_orginal

def save_data_2_csv(df, using_colors: list,all_colors: list,rating: int, luma_mean: int):#safes img and user data to csv
    #all_colors_names = [color.name for color in all_colors.colors]#Liste mit allen Farbnamen für CSV    
    #columns = ["Rating"] + ["LumaMean"] + all_colors_names #Spalten überschriften   
    used_colors = [1 if color in using_colors.colors else 0 for color in all_colors.colors]#Erstelle eine Liste, die die Verwendung der Farben speichert (0 oder 1)    
    new_data = [rating] + [luma_mean] + used_colors#Füge die Anzahl der ausgewählten Farben am Anfang der Liste hinzu
    df.loc[len(df)] = new_data    
    output_file = GROUND_PATH + "data.csv"
    df.to_csv(output_file, index=False)  
    print("--> Saved your rating and image details to data.csv file")

def calc_luma_mean(img: np.ndarray)-> int:#calculates Luma mean for an whole image
    luma_list = []   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_color = tuple(img[i, j])#returns pixel in RGB
            luma = calc_luma(pixel_color)
            luma_list.append(luma)
    luma_mean = int(np.mean(luma_list))#calc mean of the list
    #print(f"--> Luma Mean for picture is: {luma_mean}")
    luma_list.clear()#clear list for next img
    return luma_mean

def sort_df_after_best_img(luma_mean: int,df: pd.DataFrame)->pd.DataFrame:#sorts df after the highest example rating
    df["ExampleRating"] = df["Rating"]*(255-abs(luma_mean-df["LumaMean"]))
    df.sort_values(by = "ExampleRating", ascending = False, inplace = True)
    df.drop(columns=["ExampleRating"],inplace = True)
    return df   

def limit_to_range(number: int, lower_limit: int, upper_limit: int)-> int:#limits a number to a range 
    if number < lower_limit:
        return lower_limit
    elif number > upper_limit:
        return upper_limit
    else:
        return number        

def title_maker()-> str:#makes an titel with the used colors
    color_string = " ".join(color.name for color in using_colors.colors)
    title=f"Color Amount: {len(using_colors.colors)}\n Colors: {color_string}"
    return title

def adjust_old_rating(rating)->None:
    df.iloc[0]["Rating"] = int((df.iloc[0]["Rating"] + rating)/2)
    print("--> Rating for this setting was adjusted to ", df.iloc[0]["Rating"])
        
def mouse_event(event):
    if isinstance(event.xdata, (int, float)) & isinstance(event.ydata, (int, float)):
        x_pos = int(event.xdata)
        y_pos = int(event.ydata)
        if x_pos < img_art.shape[0] and y_pos < img_art.shape[1]:
            click_color_changer(x_pos,y_pos)

def find_color_by_tuple(tuple):
    for color in all_colors.colors:
        if color.rgb == tuple:
            return color
    return (0,0,0)
    
def click_color_changer(x,y):
    available_colors = [color for color in all_colors.colors if color not in using_colors.colors]
    color_at_mouse = find_color_by_tuple(tuple(img_art[x, y]))
    new_color = random.choice(available_colors)
    using_colors.remove(color_at_mouse)
    using_colors.add(new_color)
    print("--> Removed the clicked color", color_at_mouse.name, "and added" , new_color.name, "instead")
    title = title_maker()
    for i in range(img_art.shape[0]):
        for j in range(img_art.shape[1]):
            pixel_color_tuple = tuple(img_art[i, j])
            if pixel_color_tuple == color_at_mouse.rgb:
                img_art[i, j] = new_color.rgb

    update_art_img(img_art)
    
#########################################################################################################
#List with all available colors
#########################################################################################################

all_colors = ColorManager()
all_colors.add(MyColor((0, 0, 0), "Black"))
all_colors.add(MyColor((255, 255, 255), "White"))
all_colors.add(MyColor((255, 0, 0), "Red"))
all_colors.add(MyColor((0, 0, 255), "Blue"))
all_colors.add(MyColor((0, 255, 0), "Green"))
all_colors.add(MyColor((17, 216, 230), "LightBlue"))
all_colors.add(MyColor((144, 238, 14), "LightGreen"))
all_colors.add(MyColor((255, 255, 0), "Yellow"))
all_colors.add(MyColor((128, 0, 128), "Purple"))
all_colors.add(MyColor((0, 0, 139), "DarkBlue"))
all_colors.add(MyColor((255, 0, 127), "Pink"))
all_colors.add(MyColor((91, 58, 41), "Brown"))
all_colors.add(MyColor((139, 0, 0), "DarkRed"))
all_colors.add(MyColor((255, 165, 0), "Orange"))

#########################################################################################################
#Loop
#########################################################################################################    

while True:
    print("******************* NEW ART IMAGE *******************")
    df = pd.read_csv(GROUND_PATH + "data.csv")#read dataframe from csv
    img_orginal = get_random_img()#select random picture from folder: in the ground direcory has the be a folder named "Source_Images" that contains jpg files   
    luma_mean = calc_luma_mean(img_orginal)  
    df = sort_df_after_best_img(luma_mean,df)  
    #starts in column 2 and writes every color with a 1 in the list
    colors_example_name_list = [color for color in df.columns[1:] if df.iloc[0][color] == 1]
    example_colors = ColorManager()
    all_colors.add_from_names_list(example_colors, colors_example_name_list)#(where to add the colors, list with names)

    using_colors = example_colors.create_copy()
    using_colors.colors.sort(key=lambda color: color.luma)
    example_rating = df.iloc[0]["Rating"]
    using_colors.print(f"Example Image with rating {example_rating}")
    
    img_art = new_img(img_orginal, using_colors)#create art picture with the colors from example img
    axs, fig = plot_images(img_orginal, img_art)#display both pictures in one window with title and colors
    wait = input("--> Click on colors on the picture to change them, then press Enter\n")      
    rating = int(input("--> Rate the image 1-10: "))#get the rating
    
    save_data_2_csv(df, using_colors,all_colors,rating,luma_mean)
    save_img(len(using_colors.colors), rating, luma_mean)#save the plots in a jpg file
        
    user_input = (input("--> Enter for next picture 0 to end: "))#ask user if he wants to continue
    if user_input == "0":
        break
    