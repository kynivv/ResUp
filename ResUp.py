# libraries Import
from tkinter import *
import tkinter as tk
import customtkinter
from tkinter import filedialog
import cv2
from PIL import Image
import torch
import numpy as np
from Real_ESRGAN.RealESRGAN import RealESRGAN
import threading
from tkinter import ttk
import subprocess


import os
from os.path import isfile, basename


# Functions
##################################################################################
class Functions:
    displayImage_Path = "Temp/Display_Input_Image.png"
    inputImage_Path = "Temp/Input_Image.png"
    selected_model = "RealESRGAN_x4"

    model_scales = {
        "RealESRGAN_x2": 2,
        "RealESRGAN_x4": 4,
        "RealESRGAN_x8": 8,
    }

    def get_file_name(file_path):
        return basename(file_path)

    @staticmethod
    def browseImage(label):
        global filepath
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select a File",
            filetypes=(("PNG", ".png"), ("JPEG", ".jpeg"), ("JPG", ".jpg")),
        )
        global filename
        filename = Functions.get_file_name(filepath)

        image = Image.open(filepath).convert("RGB")
        image.save(Functions.inputImage_Path)
        cropped_image = image.resize((500, 500))
        cropped_image.save(Functions.displayImage_Path)

        cropped_image = tk.PhotoImage(file=Functions.displayImage_Path)
        label.configure(image=cropped_image, text="")

    def transformImage():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = RealESRGAN(device, scale=Functions.model_scales[selected_model])
        model.load_weights(f"weights/{selected_model}.pth")

        if isfile(Functions.inputImage_Path) == False:  # Image Error
            tk.messagebox.showerror(
                "Image Error", "Please pick image before transforming"
            )
            Layout.window.attributes("-disabled", False)
            loading_screen.destroy()
            exit()

        elif isfile(Functions.inputImage_Path) == True:
            pass

        image = Image.open(Functions.inputImage_Path).convert("RGB")

        sr_image = model.predict(image)

        sr_image.save(f"Results/{selected_model}_{filename}")

        # Deleting Temp Files
        os.remove(Functions.displayImage_Path)
        os.remove(Functions.inputImage_Path)
        Layout.Input_Image_Label.configure(image="", text="")
        Layout.Input_Image_Label.image = None

        # Close the loading screen
        Functions.close_loading_screen()
        tk.messagebox.showinfo(
            "Success",
            "The image was upscaled successfully!\n\nPlease select a new image before the transformation",
        )
        Layout.Input_Image_Label.configure(image=None, text="")

    @staticmethod
    def show_loading_screen():
        global loading_screen
        loading_screen = Toplevel(Layout.window)
        loading_screen.title("Image processing")
        loading_screen.iconbitmap("ResUp.ico")
        loading_screen.geometry("350x150")
        loading_screen.resizable(0, 0)

        # Add a label to the loading screen
        label = Label(loading_screen, text="Please wait, processing...")
        label.pack(pady=20)

        # Add a progress bar
        progress = ttk.Progressbar(loading_screen, mode="indeterminate")
        progress.pack(pady=20)
        progress.start()

        # Center the loading screen
        loading_screen.update_idletasks()
        width = loading_screen.winfo_width()
        height = loading_screen.winfo_height()
        x = (loading_screen.winfo_screenwidth() // 2) - (width // 2)
        y = (loading_screen.winfo_screenheight() // 2) - (height // 2)
        loading_screen.geometry(f"{width}x{height}+{x}+{y}")

        # Disable the main window
        Layout.window.attributes("-disabled", True)

    @staticmethod
    def close_loading_screen():
        loading_screen.destroy()
        # Re-enable the main window
        Layout.window.attributes("-disabled", False)

    @staticmethod
    def start_transform_image():
        # Show loading screen
        Functions.show_loading_screen()

        # Run transformImage in a separate thread
        threading.Thread(target=Functions.transformImage).start()

    def FileLoc():
        absolute_path = os.path.abspath(f"Results/")
        subprocess.Popen(["explorer", absolute_path])

    def on_combobox_select(event):
        global selected_model
        selected_model = event.widget.get()
        print(f"Model: {selected_model} selected")

    @staticmethod
    def post_mainloop_action():
        if isfile(Functions.inputImage_Path) == True:
            os.remove(Functions.inputImage_Path)
        if isfile(Functions.displayImage_Path) == True:
            os.remove(Functions.displayImage_Path)

    @staticmethod
    def on_closing():
        print("Closing the application...")
        Functions.post_mainloop_action()
        Layout.window.destroy()


# Layout
##################################################################################
class Layout:
    window = Tk()
    window.iconbitmap("ResUp.ico")
    window.title("ResUp")
    window.geometry("900x650")
    window.resizable(0, 0)
    window.configure(bg="#002451")

    window.protocol("WM_DELETE_WINDOW", Functions.on_closing)  # Closing Protocol

    combo = ttk.Combobox(
        master=window,
        state="readonly",
        font=("Segoe UI Semibold", 8),
        width=15,
        values=["RealESRGAN_x2", "RealESRGAN_x4", "RealESRGAN_x8"],
        style="TCombobox",
    )
    default_value = "RealESRGAN_x4"
    combo.set(default_value)
    global selected_model
    selected_model = "RealESRGAN_x4"
    print(f"Model: {selected_model} selected")
    combo.bind("<<ComboboxSelected>>", Functions.on_combobox_select)
    combo.place(x=40, y=20)

    Input_Image_Label = customtkinter.CTkLabel(
        master=window,
        text="",
        font=("Segoe UI Semibold", 18),
        text_color="#eaebed",
        height=500,
        width=500,
        corner_radius=0,
        bg_color="#eaebed",
        fg_color="#002451",
    )
    Input_Image_Label.place(x=200, y=0)

    Input_Button = customtkinter.CTkButton(
        master=window,
        text="Browse Image",
        font=("Segoe UI Semibold", 14),
        text_color="#eaebed",
        hover=True,
        hover_color="#00377a",
        height=35,
        width=105,
        border_width=3,
        corner_radius=16,
        border_color="#eaebed",
        bg_color="#002451",
        fg_color="#002451",
        command=lambda: Functions.browseImage(Layout.Input_Image_Label),
    )
    Input_Button.place(x=200, y=520)

    Transform_Button = customtkinter.CTkButton(
        master=window,
        text="Upscale Image",
        font=("Segoe UI Semibold", 14),
        text_color="#eaebed",
        hover=True,
        hover_color="#00377a",
        height=35,
        width=105,
        border_width=3,
        corner_radius=16,
        border_color="#eaebed",
        bg_color="#002451",
        fg_color="#002451",
        command=lambda: Functions.start_transform_image(),
    )
    Transform_Button.place(x=350, y=520)

    File_Loc_Button = customtkinter.CTkButton(
        master=window,
        text="Output Folder",
        font=("Segoe UI Semibold", 14),
        text_color="#eaebed",
        hover=True,
        hover_color="#00377a",
        height=35,
        width=105,
        border_width=3,
        corner_radius=16,
        border_color="#eaebed",
        bg_color="#002451",
        fg_color="#002451",
        command=lambda: Functions.FileLoc(),
    )
    File_Loc_Button.place(x=500, y=520)


# Main Loop
##################################################################################
Layout().window.mainloop()
