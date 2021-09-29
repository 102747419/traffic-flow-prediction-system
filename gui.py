import tkinter as tk


def main():
    print('running main')
    window = tk.Tk()

    # define widgets
    canvas = tk.Canvas(window, width=600, height=500)
    canvas.pack()

    # right_offset, Start, length, end
    # canvas.create_line(0, 100, 600, 100)

    lbl_heading = tk.Label(window, text='Traffic Flow Prediction')
    lbl_heading.config(font=('helvetica', 20))
    lbl_time = tk.Label(window, text='Departure Time: ')
    txt_time = tk.Entry(window)
    btn_exit = tk.Button(window, text='exit', command=window.destroy)
    btn_calculate = tk.Button(window, text='Calculate Route', command=calc_route(txt_time.get()))

    # render widgets
    canvas.create_window(300, 20, window=lbl_heading)
    canvas.create_window(150, 60, window=lbl_time)
    canvas.create_window(450, 60, window=txt_time)
    canvas.create_window(150, 200, window=btn_exit)
    canvas.create_window(450, 200, window=btn_calculate)
    window.mainloop()


def calc_route(time):
    print('Calc route: ', time)
    return 0


main()
