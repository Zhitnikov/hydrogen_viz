import sys
import numpy as np
from core.physics import probability_density
from core.grid import generate_grid
from viz.plotter import create_orbital_figure

def get_quantum_numbers():
    print("=== Конфигурация ===")
    while True:
        try:
            n = int(input("Введите главное квантовое число n (n >= 1): "))
            if n < 1:
                print("n должно быть >= 1.")
                continue
            
            l = int(input(f"Введите орбитальное квантовое число l (0 <= l < {n}): "))
            if l < 0 or l >= n:
                print(f"l должно быть между 0 и {n-1}.")
                continue
                
            m = int(input(f"Введите магнитное квантовое число m ({-l} <= m <= {l}): "))
            if m < -l or m > l:
                print(f"m должно быть между {-l} и {l}.")
                continue
                
            return n, l, m
        except ValueError:
            print("Ошибка: введите целое число.")

def main():
    print("Визуализатор орбиталей водорода")
    print("--------------------------------------")
    
    while True:
        n, l, m = get_quantum_numbers()
        
        print(f"\nРасчет орбитали для n={n}, l={l}, m={m}...")
        
        extent = 3.0 * (n**2)
        resolution = 50 
        
        print(f"Генерация сетки (размер +/-{extent:.1f} a0, разрешение {resolution})...")
        X, Y, Z, R, Theta, Phi = generate_grid(extent, resolution)
        
        print("Расчет волновых функций...")
        prob_density = probability_density(n, l, m, R, Theta, Phi)
        
        print("Создание 3D модели...")
        fig = create_orbital_figure(prob_density, X, Y, Z, n, l, m)
        
        print("Открытие визуализации в браузере...")
        fig.show()
        
        cont = input("\nВизуализировать другую орбиталь? (y/n): ").lower()
        if cont != 'y':
            break

    print("Выход.")

if __name__ == "__main__":
    main()
