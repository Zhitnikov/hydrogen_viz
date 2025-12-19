import tkinter as tk
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import sys
import os

from core.physics import probability_density
from core.grid import generate_grid
from viz.plotter import create_orbital_figure_matplotlib, create_orbital_figure

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Визуализатор орбиталей водорода")
        self.geometry("1100x750")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.home_button = ctk.CTkButton(self.sidebar, text="🏠 Главная", command=self.show_menu, 
                                          fg_color="transparent", text_color=("gray10", "gray90"),
                                          hover_color=("gray70", "gray30"), width=100)
        self.home_button.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nw")

        self.logo_label = ctk.CTkLabel(self.sidebar, text="Orbital Engine", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=1, column=0, padx=20, pady=(20, 30))

        self.n_label = ctk.CTkLabel(self.sidebar, text="Главное число (n):")
        self.n_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.n_entry = ctk.CTkEntry(self.sidebar, placeholder_text="напр. 1")
        self.n_entry.insert(0, "1")
        self.n_entry.grid(row=3, column=0, padx=20, pady=(0, 10))

        self.l_label = ctk.CTkLabel(self.sidebar, text="Орбитальное число (l):")
        self.l_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")
        self.l_entry = ctk.CTkEntry(self.sidebar, placeholder_text="напр. 0")
        self.l_entry.insert(0, "0")
        self.l_entry.grid(row=5, column=0, padx=20, pady=(0, 10))

        self.m_label = ctk.CTkLabel(self.sidebar, text="Магнитное число (m):")
        self.m_label.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
        self.m_entry = ctk.CTkEntry(self.sidebar, placeholder_text="напр. 0")
        self.m_entry.insert(0, "0")
        self.m_entry.grid(row=7, column=0, padx=20, pady=(0, 20))

        self.calc_button = ctk.CTkButton(self.sidebar, text="Визуализировать", command=self.on_visualize)
        self.calc_button.grid(row=8, column=0, padx=20, pady=(10, 5))

        self.browser_button = ctk.CTkButton(self.sidebar, text="Открыть в браузере", 
                                             command=self.on_browser_visualize,
                                             fg_color="transparent", border_width=2)
        self.browser_button.grid(row=9, column=0, padx=20, pady=5)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Готов", text_color="gray")
        self.status_label.grid(row=10, column=0, padx=20, pady=20)

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        self.cards = []

        self.menu_frame = ctk.CTkScrollableFrame(self.content_frame, label_text="Быстрый выбор орбиталей", label_font=ctk.CTkFont(size=18, weight="bold"))
        self.menu_frame.grid(row=0, column=0, sticky="nsew")
        self.setup_menu_cards()

        self.plot_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        
        self.canvas = None
        self.toolbar = None
        
        self.show_menu()

    def setup_menu_cards(self):
        presets = [
            {"name": "1s Основное состояние", "n": 1, "l": 0, "m": 0, "color": "#FF5733"},
            {"name": "2p Полярное состояние", "n": 2, "l": 1, "m": 0, "color": "#33FF57"},
            {"name": "3d Сложная", "n": 3, "l": 2, "m": 0, "color": "#3357FF"},
            {"name": "4f Продвинутая", "n": 4, "l": 3, "m": 0, "color": "#F033FF"},
            {"name": "2p (m=1) Состояние", "n": 2, "l": 1, "m": 1, "color": "#FFD433"},
            {"name": "3p Орбиталь", "n": 3, "l": 1, "m": 0, "color": "#33FFF0"},
            {"name": "3d (m=2) Состояние", "n": 3, "l": 2, "m": 2, "color": "#A133FF"},
            {"name": "4d Орбиталь", "n": 4, "l": 2, "m": 0, "color": "#FF338A"},
            {"name": "5g Теоретическая", "n": 5, "l": 4, "m": 0, "color": "#33FFA1"}
        ]

        cols = 3
        for i, p in enumerate(presets):
            card = ctk.CTkFrame(self.menu_frame, corner_radius=12, border_width=1, border_color="gray50")
            card.grid(row=i//cols, column=i%cols, padx=10, pady=10, sticky="nsew")
            self.cards.append(card)
            
            select_cmd = lambda e, params=p, current_card=card: self.select_preset(params, current_card)
            
            card.bind("<Button-1>", select_cmd)
            
            preview = ctk.CTkFrame(card, width=220, height=100, fg_color=p["color"], corner_radius=8)
            preview.pack(padx=15, pady=(15, 5))
            preview.bind("<Button-1>", select_cmd)
            
            label = ctk.CTkLabel(card, text=p["name"], font=ctk.CTkFont(size=14, weight="bold"))
            label.pack(padx=15, pady=2)
            label.bind("<Button-1>", select_cmd)
            
            numbers = ctk.CTkLabel(card, text=f"n={p['n']}, l={p['l']}, m={p['m']}", font=ctk.CTkFont(size=12))
            numbers.pack(padx=15, pady=(0, 15))
            numbers.bind("<Button-1>", select_cmd)

    def select_preset(self, p, selected_card):
        for card in self.cards:
            card.configure(border_color="gray50", border_width=1)
        
        selected_card.configure(border_color="#1f538d", border_width=3)

        self.n_entry.delete(0, tk.END)
        self.n_entry.insert(0, str(p["n"]))
        self.l_entry.delete(0, tk.END)
        self.l_entry.insert(0, str(p["l"]))
        self.m_entry.delete(0, tk.END)
        self.m_entry.insert(0, str(p["m"]))
        
        self.status_label.configure(text=f"Выбрано: {p['name']}", text_color="white", font=ctk.CTkFont(size=12))

    def show_menu(self):
        self.plot_frame.grid_forget()
        self.menu_frame.grid(row=0, column=0, sticky="nsew")
        self.status_label.configure(text="Готов")

    def show_plot(self):
        self.menu_frame.grid_forget()
        self.plot_frame.grid(row=0, column=0, sticky="nsew")

    def on_visualize(self):
        try:
            n = int(self.n_entry.get())
            l = int(self.l_entry.get())
            m = int(self.m_entry.get())

            if n < 1: raise ValueError("n должно быть >= 1")
            if not (0 <= l < n): raise ValueError(f"l должно быть между 0 и {n-1}")
            if not (-l <= m <= l): raise ValueError(f"m должно быть между {-l} и {l}")

            self.show_plot()
            self.status_label.configure(text="Вычисление...", text_color="orange")
            self.calc_button.configure(state="disabled")

            threading.Thread(target=self.calculate, args=(n, l, m), daemon=True).start()

        except ValueError as e:
            tk.messagebox.showerror("Ошибка ввода", str(e))

    def calculate(self, n, l, m):
        try:
            extent = 3.0 * (n**2)
            resolution = 55
            
            X, Y, Z, R, Theta, Phi = generate_grid(extent, resolution)
            prob_density = probability_density(n, l, m, R, Theta, Phi)
            
            data = {
                'density': prob_density,
                'X': X, 'Y': Y, 'Z': Z,
                'n': n, 'l': l, 'm': m
            }
            
            self.after(0, self.update_plot, data)
        except Exception as e:
            self.after(0, lambda: tk.messagebox.showerror("Ошибка", str(e)))
            self.after(0, self.reset_ui)

    def update_plot(self, data):
        fig = create_orbital_figure_matplotlib(
            data['density'], data['X'], data['Y'], data['Z'], 
            data['n'], data['l'], data['m']
        )

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")

        self.reset_ui()
        self.status_label.configure(text="Готово", text_color="green")

    def reset_ui(self):
        self.calc_button.configure(state="normal")

    def on_browser_visualize(self):
        try:
            n = int(self.n_entry.get())
            l = int(self.l_entry.get())
            m = int(self.m_entry.get())

            if n < 1: raise ValueError("n должно быть >= 1")
            if not (0 <= l < n): raise ValueError(f"l должно быть между 0 и {n-1}")
            if not (-l <= m <= l): raise ValueError(f"m должно быть между {-l} и {l}")

            self.status_label.configure(text="Открываю браузер...", text_color="orange")
            
            threading.Thread(target=self.calculate_browser, args=(n, l, m), daemon=True).start()

        except ValueError as e:
            tk.messagebox.showerror("Ошибка ввода", str(e))

    def calculate_browser(self, n, l, m):
        try:
            extent = 3.0 * (n**2)
            resolution = 50
            
            X, Y, Z, R, Theta, Phi = generate_grid(extent, resolution)
            prob_density = probability_density(n, l, m, R, Theta, Phi)
            
            fig = create_orbital_figure(prob_density, X, Y, Z, n, l, m)
            fig.show()
            
            self.after(0, lambda: self.status_label.configure(text="Браузер открыт", text_color="green"))
        except Exception as e:
            self.after(0, lambda: tk.messagebox.showerror("Ошибка", str(e)))

if __name__ == "__main__":
    app = App()
    app.mainloop()
