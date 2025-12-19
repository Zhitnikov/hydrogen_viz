import PyInstaller.__main__
import os
import customtkinter

def build():
    ctk_path = os.path.dirname(customtkinter.__file__)
    
    print("Начало сборки EXE...")
    
    PyInstaller.__main__.run([
        'gui_main.py',
        '--onefile',
        '--noconsole',
        f'--add-data={ctk_path};customtkinter/',
        '--name=HydrogenVisualizer',
        '--clean'
    ])
    
    print("\nСборка завершена. Проверьте папку 'dist'.")

if __name__ == "__main__":
    build()
