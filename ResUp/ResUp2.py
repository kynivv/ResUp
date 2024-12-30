from multiprocessing import Process, Queue
import threading
import tkinter as tk
from tkinter import filedialog, Toplevel, ttk
from tkinter import messagebox
from os.path import isfile, basename
from PIL import Image, ImageTk
from Real_ESRGAN.RealESRGAN import RealESRGAN
import numpy as np
import subprocess
import torch
import customtkinter
import os
import time
from AnimeGANv3 import AnimeGANv3
from pathlib import Path

import onnxruntime as ort
import onnx
import imageio


class Functions:
    displayImage_Path = "Temp/Display/Display_Input_Image.png"
    inputImage_Path = "Temp/Input/Input_Image.png"
    inputImageFolder_Path = "Temp/Input"
    selected_model = "RealESRGAN_x4"
    LOGO_Path = "ResUpLOGOPNG.png"

    model_scales = {
        "RealESRGAN_x2": 2,
        "RealESRGAN_x4": 4,
        "RealESRGAN_x8": 8,
        "AnimeGANv3_H40_x1": "H40",
        "AnimeGANv3_H50_x1": "H50",
        "AnimeGANv3_H64_x1": "H64",
        "UltraSharp_x4": 4,
    }

    @staticmethod
    def get_file_name(file_path):
        return basename(file_path)

    @staticmethod
    def show_success_message():  # Success message
        messagebox.showinfo(
            "Success",
            "The image was upscaled successfully!\n\nPlease select a new image before transformation",
        )

    @staticmethod
    def AnimeGan_processing():
        AnimeGANv3.TransformImage(
            Path(Functions.inputImageFolder_Path),
            Path(f"Results"),
            (Functions.model_scales[selected_model]),
        )

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

        cropped_image = ImageTk.PhotoImage(file=Functions.displayImage_Path)
        label.configure(image=cropped_image, text="")
        label.image = cropped_image  # Keep a reference to avoid garbage collection

    @staticmethod
    def transformImage():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not isfile(Functions.inputImage_Path):  # Image Error
            tk.messagebox.showerror(
                "Image Error", "Please pick an image before transforming"
            )
            Layout.window.attributes("-disabled", False)
            Functions.close_loading_screen()
            return

        elif isfile(Functions.inputImage_Path):
            if "RealESRGAN" in selected_model:
                model = RealESRGAN(device, scale=Functions.model_scales[selected_model])
                model.load_weights(f"weights/{selected_model}.pth")
                image = Image.open(Functions.inputImage_Path).convert("RGB")
                sr_image = model.predict(image)
                sr_image.save(f"Results/{selected_model}_{filename}")

            elif "AnimeGANv3" in selected_model:
                Functions.AnimeGan_processing()
                os.rename(
                    "Results/Input_Image.png", f"Results/{selected_model}_{filename}"
                )

            elif "UltraSharp" in selected_model:
                model_path = f"weights/{selected_model}.onnx"
                session = ort.InferenceSession(model_path)
                model = onnx.load(model_path)

                # Get the model's input and output information
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                image = Image.open(Functions.inputImage_Path).convert("RGB")
                image = np.transpose(image, (2, 0, 1))
                input_data = np.array(image).astype(np.float32)
                input_data = np.expand_dims(input_data, axis=0)

                outputs = session.run([output_name], {input_name: input_data})
                output_image = outputs[0]

                # Reshape and squeeze the output array
                output_image = output_image.squeeze()
                output_image = np.transpose(output_image, (1, 2, 0))

                # Normalize
                output_image = np.clip(output_image, 0, 255)
                output_image = np.round(output_image)
                output_image = output_image.astype(np.uint8)

                imageio.imwrite(f"Results/{selected_model}_{filename}", output_image)

            # Clean up temporary files
            os.remove(Functions.displayImage_Path)
            os.remove(Functions.inputImage_Path)
            Layout.Input_Image_Label.configure(image="", text="")
            Layout.Input_Image_Label.image = None
            Functions.close_loading_screen()
            time.sleep(0.1)
            Functions.show_success_message()

    @staticmethod
    def show_loading_screen():
        global loading_screen
        loading_screen = Toplevel(Layout.window)
        loading_screen.title("Image processing")
        loading_screen.iconbitmap("ResUp.ico")
        loading_screen.geometry("350x150")
        loading_screen.resizable(0, 0)

        label = tk.Label(loading_screen, text="Please wait, processing...")
        label.pack(pady=20)

        progress = ttk.Progressbar(loading_screen, mode="indeterminate")
        progress.pack(pady=20)
        progress.start()

        loading_screen.update_idletasks()
        width = loading_screen.winfo_width()
        height = loading_screen.winfo_height()
        x = (loading_screen.winfo_screenwidth() // 2) - (width // 2)
        y = (loading_screen.winfo_screenheight() // 2) - (height // 2)
        loading_screen.geometry(f"{width}x{height}+{x}+{y}")

        Layout.window.attributes("-disabled", True)

    @staticmethod
    def close_loading_screen():
        if "loading_screen" in globals():
            loading_screen.destroy()
        Layout.window.attributes("-disabled", False)

    @staticmethod
    def start_transform_image():
        Functions.show_loading_screen()
        threading.Thread(target=Functions.transformImage).start()

    @staticmethod
    def FileLoc():
        absolute_path = os.path.abspath(f"Results/")
        subprocess.Popen(["explorer", absolute_path])

    @staticmethod
    def on_combobox_select(event):
        global selected_model
        selected_model = event
        print(selected_model, " is Selected")

    @staticmethod
    def post_mainloop_action():
        if isfile(Functions.inputImage_Path):
            os.remove(Functions.inputImage_Path)
        if isfile(Functions.displayImage_Path):
            os.remove(Functions.displayImage_Path)

    @staticmethod
    def on_closing():
        Functions.post_mainloop_action()
        Layout.window.destroy()


class Layout:
    window = tk.Tk()
    window.iconbitmap("ResUp.ico")
    window.title("ResUp")
    window.geometry("900x650")
    window.resizable(0, 0)
    window.configure(bg="#02202C")

    window.protocol("WM_DELETE_WINDOW", Functions.on_closing)

    combo = customtkinter.CTkComboBox(
        master=window,
        state="readonly",
        font=("Segoe UI Semibold", 10),
        width=135,
        height=25,
        values=[
            "RealESRGAN_x2",
            "RealESRGAN_x4",
            "RealESRGAN_x8",
            "AnimeGANv3_H40_x1",
            "AnimeGANv3_H50_x1",
            "AnimeGANv3_H64_x1",
            "UltraSharp_x4",
        ],
        fg_color="#02202C",
        text_color="#34E4EA",
        border_color="#34E4EA",
        border_width=3,
        dropdown_text_color="#34E4EA",
        dropdown_fg_color="#02202C",
        button_color="#005259",
        command=Functions.on_combobox_select,
        dropdown_hover_color="#005259",
        dropdown_font=("Segoe UI Semibold", 10),
        justify="left",
        corner_radius=16,
    )
    combo.set("RealESRGAN_x4")
    combo.place(x=30, y=30)

    Input_Image_frame = customtkinter.CTkFrame(
        master=window,
        width=506,  # Width slightly larger than the label's width
        height=506,  # Height slightly larger than the label's height
        corner_radius=0,
        bg_color="#34E4EA",
        fg_color="#34E4EA",  # Stroke color
    )
    Input_Image_frame.place(x=197, y=-3)  # Adjusted to fit the stroke position

    Input_Image_Label = customtkinter.CTkLabel(
        master=window,
        text="",
        font=("Segoe UI Semibold", 14),
        text_color="#eaebed",
        height=500,
        width=500,
        corner_radius=0,
        bg_color="#eaebed",
        fg_color="#02202C",
    )
    Input_Image_Label.place(x=200, y=0)

    Input_Button = customtkinter.CTkButton(
        master=window,
        text="Browse Image",
        font=("Segoe UI Semibold", 14),
        text_color="#34E4EA",
        hover=True,
        hover_color="#005259",
        height=35,
        width=105,
        border_width=3,
        corner_radius=16,
        border_color="#34E4EA",
        bg_color="#02202C",
        fg_color="#02202C",
        command=lambda: Functions.browseImage(Layout.Input_Image_Label),
    )
    Input_Button.place(x=200, y=520)

    Transform_Button = customtkinter.CTkButton(
        master=window,
        text="Upscale Image",
        font=("Segoe UI Semibold", 14),
        text_color="#34E4EA",
        hover=True,
        hover_color="#005259",
        height=35,
        width=105,
        border_width=3,
        corner_radius=16,
        border_color="#34E4EA",
        bg_color="#02202C",
        fg_color="#02202C",
        command=lambda: Functions.start_transform_image(),
    )
    Transform_Button.place(x=350, y=520)

    File_Loc_Button = customtkinter.CTkButton(
        master=window,
        text="Output Folder",
        font=("Segoe UI Semibold", 14),
        text_color="#34E4EA",
        hover=True,
        hover_color="#005259",
        height=35,
        width=105,
        border_width=3,
        corner_radius=16,
        border_color="#34E4EA",
        bg_color="#02202C",
        fg_color="#02202C",
        command=lambda: Functions.FileLoc(),
    )
    File_Loc_Button.place(x=500, y=520)

    # LOGO PLacement
    OriginalLOGO = Image.open(Functions.LOGO_Path)
    ResizedLOGO = OriginalLOGO.resize((65, 65))
    TKLOGO = ImageTk.PhotoImage(ResizedLOGO)
    LOGO_label = customtkinter.CTkLabel(
        window, text="", image=TKLOGO, bg_color="#02202C"
    )
    LOGO_label.image = TKLOGO
    LOGO_label.place(x=816, y=566)


if __name__ == "__main__":
    Layout().window.mainloop()
