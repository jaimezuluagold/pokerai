import pyautogui

print("Mueve el ratón a la esquina SUPERIOR IZQUIERDA de la mesa de póker")
print("Tienes 5 segundos...")
import time
time.sleep(5)

x1, y1 = pyautogui.position()
print(f"Esquina superior izquierda: {x1}, {y1}")

print("Ahora mueve el ratón a la esquina INFERIOR DERECHA de la mesa de póker")
print("Tienes 5 segundos...")
time.sleep(5)

x2, y2 = pyautogui.position()
print(f"Esquina inferior derecha: {x2}, {y2}")

# Calcular dimensiones
width = x2 - x1
height = y2 - y1

print("\nCoordenadas para el código:")
print(f"manual_game_region = {{")
print(f"    \"top\": {y1},")
print(f"    \"left\": {x1},")
print(f"    \"width\": {width},")
print(f"    \"height\": {height}")
print(f"}}")

input("Presiona Enter para salir...")