import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import loading
import pandas as pd
import matplotlib.dates as mdates


# List of functions and their corresponding names
data_complex = loading.Data("data_complex")
all_aircrafts = list(data_complex.aircraft["registration"])
print(all_aircrafts)
airport_to_index = {code:i for i, code in enumerate(data_complex.airports["code"])}
print(airport_to_index)
"""
functions = [
    {"name": "Sine Wave", "func": lambda x, amp, freq: amp * np.sin(freq * x)},
    {"name": "Cosine Wave", "func": lambda x, amp, freq: amp * np.cos(freq * x)},
    {"name": "Tangent Wave", "func": lambda x, amp, freq: amp * np.tan(freq * x)},
    {"name": "Exponential Decay", "func": lambda x, amp, freq: amp * np.exp(-freq * x)},
    {"name": "Logarithmic Function", "func": lambda x, amp, freq: amp * np.log(x + 1)}
]"""



# Function to update the plot based on selected checkboxes
def update_plot():
    # Get values from sliders
    amplitude = amp_slider.get()
    frequency = freq_slider.get()

    # Clear the previous plot
    ax.clear()

    range_start =pd.to_datetime("2008-03-01 0:00")
    range_end = pd.to_datetime("2008-03-01 23:59")
    # Generate x values
    ax.vlines(x=range_start, ymin = 0 , ymax=len(airport_to_index), color="black")
    ax.vlines(x=range_end, ymin = 0 , ymax=len(airport_to_index), color="black")

    # Plot selected functions
    aircraft_color_id = [0]*len(all_aircrafts)
    count_of_labeled = 0
    was_airport_drawn = [False]*len(airport_to_index)
    for i,row in data_complex.flights.iterrows():
        aircraft_pos = all_aircrafts.index(row["aircraft"])
        if  func_vars[aircraft_pos].get():
            if 0 == aircraft_color_id[aircraft_pos]:
                count_of_labeled += 1
                aircraft_color_id[aircraft_pos] = count_of_labeled
                ax.plot([row["start"], row["finish"]], [airport_to_index[row["from"]],airport_to_index[row["to"]]], 
                        color=["red", "green", "blue", "lightblue"][count_of_labeled % 4], label=row["aircraft"] )
            else:
                ax.plot([row["start"], row["finish"]], [airport_to_index[row["from"]],airport_to_index[row["to"]]], 
                        color=["red", "green", "blue", "lightblue"][aircraft_color_id[aircraft_pos] % 4])

            diff = (range_end - range_start)
            if not was_airport_drawn[airport_to_index[row["from"]]]:
                ax.hlines(y=airport_to_index[row["from"]], xmin=range_start, xmax=range_end, color="black", linewidths=0.5)
                ax.annotate(row["from"], (range_start + (diff/15) * (airport_to_index[row["from"]] % 5), airport_to_index[row["from"]]  ))
            was_airport_drawn[airport_to_index[row["from"]]] = True

            if not was_airport_drawn[airport_to_index[row["to"]]]:
                ax.hlines(y=airport_to_index[row["to"]], xmin=range_start, xmax=range_end, color="black", linewidths=0.5)
                ax.annotate(row["to"], (range_start + (diff/15) * (airport_to_index[row["to"]] % 5), airport_to_index[row["to"]]  ))
            was_airport_drawn[airport_to_index[row["to"]]] = True

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Adjust interval for your case
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
    plt.gcf().autofmt_xdate()
    
    # Redraw the plot
    ax.legend()
    canvas.draw()


# Create tkinter window
root = tk.Tk()
root.title("Plot Multiple Functions with Checkboxes")


# Create figure and axis for the plot
fig, ax = plt.subplots()

# Create the matplotlib canvas and embed it in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Sliders for amplitude and frequency
amp_slider = tk.Scale(root, from_=0.1, to=5, resolution=0.1, orient=tk.HORIZONTAL, label="Amplitude", command=lambda x: update_plot())
amp_slider.set(1)
amp_slider.pack(side=tk.TOP)

freq_slider = tk.Scale(root, from_=0.1, to=5, resolution=0.1, orient=tk.HORIZONTAL, label="Frequency", command=lambda x: update_plot())
freq_slider.set(1)
freq_slider.pack(side=tk.TOP)


# Dynamically generate checkboxes for each function in the list

func_vars = []
class Example(tk.Frame):
    def __init__(self, parent):

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        self.frame = tk.Frame(self.canvas, background="#ffffff")
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw",
                                  tags="self.frame")

        self.frame.bind("<Configure>", self.onFrameConfigure)

        self.populate()

    def populate(self):
        '''Put in some fake data'''
        for name in all_aircrafts:
            var = tk.BooleanVar(value=False)
            func_vars.append(var)
            chk = tk.Checkbutton(self.frame, text=name, variable=var, command=update_plot)
            chk.pack(side=tk.TOP)


    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


checkBoxes = Example(root)
checkBoxes.pack(side = tk.RIGHT)

# Initialize plot with default values
update_plot()

# Start tkinter main loop
root.mainloop()
