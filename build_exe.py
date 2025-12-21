import os
import subprocess
import sys

def run_command(command):
    print(f"Выполнение: {command}")
    process = subprocess.Popen(command, shell=True)
    process.communicate()
    if process.returncode != 0:
        print(f"Ошибка при выполнении: {command}")
        sys.exit(1)

def build():
    venv_dir = "venv_build"
    python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    pyinstaller_exe = os.path.join(venv_dir, "Scripts", "pyinstaller.exe")

    if not os.path.exists(venv_dir):
        print(f"Создание виртуального окружения {venv_dir}...")
        run_command(f"{sys.executable} -m venv {venv_dir}")
        
    print("Установка зависимостей...")
    run_command(f"{python_exe} -m pip install -r requirements.txt")

    import_cmd = f"{python_exe} -c \"import customtkinter; import os; print(os.path.dirname(customtkinter.__file__))\""
    ctk_path = subprocess.check_output(import_cmd, shell=True).decode().strip()
    
    print(f"Путь customtkinter: {ctk_path}")
    print("Начало сборки EXE...")
    
    build_cmd = [
        pyinstaller_exe,
        'gui_main.py',
        '--onefile',
        '--noconsole',
        f'--add-data={ctk_path}{os.pathsep}customtkinter/',
        '--name=HydrogenVisualizer',
        '--clean'
    ]
    
    run_command(" ".join(build_cmd))
    
    print("\nСборка завершена. Проверьте папку 'dist'.")

if __name__ == "__main__":
    build()
