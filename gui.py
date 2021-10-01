import tkinter as tk

# Use 'pip3 install tk' to install library


def main():
    print('running main')
    window = tk.Tk()

    # define widgets
    canvas = tk.Canvas(window, width=600, height=500)
    canvas.pack()

    INTERSECTIONS = [
        "964 ABBOTTS/CLELANDS DEVELOPMENTS",
        "968,ABBOTTS/GAINE/MONASH",
        "972,ABBOTTS/NATIONAl",
        "983,ABBOTTS/REMINGTON",
    ]

    HOURS = []
    for i in range(0, 24):
        HOURS.append(i)

    MINUTES = [0, 15, 30, 45]

    hours = tk.IntVar(window)
    hours.set(HOURS[0])  # default value
    minutes = tk.IntVar(window)
    minutes.set(MINUTES[0])  # default value
    start = tk.StringVar(window)
    start.set(INTERSECTIONS[0])  # default value
    end = tk.StringVar(window)
    end.set(INTERSECTIONS[0])  # default value

    # right_offset, Start, length, end
    # canvas.create_line(0, 100, 600, 100)

    lbl_heading = tk.Label(window, text='Traffic Flow Prediction')
    lbl_heading.config(font=('helvetica', 20))
    lbl_time = tk.Label(window, text='Departure Time: ')
    # txt_time = tk.Entry(window)
    drp_hour = tk.OptionMenu(window, hours, *HOURS)
    drp_minute = tk.OptionMenu(window, minutes, *MINUTES)
    lbl_start = tk.Label(window, text='Starting Location: ')
    drp_start = tk.OptionMenu(window, start, *INTERSECTIONS)
    lbl_end = tk.Label(window, text='Destination: ')
    drp_end = tk.OptionMenu(window, end, *INTERSECTIONS)
    btn_exit = tk.Button(window, text='exit', command=window.destroy)
    btn_calculate = tk.Button(window, text='Calculate Route', command=lambda: calc_route(int((hours.get() * 4) + (minutes.get() / 15)), start.get(), end.get()))

    # render widgets
    canvas.create_window(300, 20, window=lbl_heading)
    canvas.create_window(150, 60, window=lbl_time)
    # canvas.create_window(450, 60, window=txt_time)
    canvas.create_window(400, 60, window=drp_hour)
    canvas.create_window(490, 60, window=drp_minute)
    canvas.create_window(150, 100, window=lbl_start)
    canvas.create_window(450, 100, window=drp_start)
    canvas.create_window(150, 140, window=lbl_end)
    canvas.create_window(450, 140, window=drp_end)
    canvas.create_window(150, 200, window=btn_exit)
    canvas.create_window(450, 200, window=btn_calculate)
    window.mainloop()


def calc_route(time, start, end):
    print(f'Leaving at {time}, from {start} to {end}')


main()
