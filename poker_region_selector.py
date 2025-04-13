"""
Poker Region Selector Tool - Mejorado

Este script permite seleccionar con precisión todas las regiones de la mesa de póker.
"""

import cv2
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import messagebox, simpledialog
import os

# Variables globales
regions = {}
region_order = [
    "poker_table",  # Primero la mesa completa
    "community_cards",
    "player_cards",
    "pot",
    "player_balance",
    "fold_button",
    "check_call_button",
    "raise_button"
]
current_region_index = 0
start_point = None
end_point = None
image = None

def take_screenshot():
    """Tomar una captura de pantalla completa."""
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def mouse_callback(event, x, y, flags, param):
    """Manejar eventos del mouse para selección de regiones."""
    global start_point, end_point, image, current_region_index
    
    img_copy = image.copy()
    
    # Dibujar las regiones ya existentes
    for name, coords in regions.items():
        cv2.rectangle(img_copy, 
                     (coords['left'], coords['top']), 
                     (coords['left'] + coords['width'], coords['top'] + coords['height']),
                     (0, 255, 0), 2)
        cv2.putText(img_copy, name, (coords['left'], coords['top'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Mostrar instrucciones
    current_region = region_order[current_region_index] if current_region_index < len(region_order) else "DONE"
    instruction = f"Selecciona la región: {current_region} ({current_region_index+1}/{len(region_order)})"
    cv2.putText(img_copy, instruction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        end_point = (x, y)
        # Dibujar rectángulo
        cv2.rectangle(img_copy, start_point, end_point, (0, 0, 255), 2)
    
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        
        if current_region_index < len(region_order):
            region_name = region_order[current_region_index]
            
            # Calcular coordenadas
            left = min(start_point[0], end_point[0])
            top = min(start_point[1], end_point[1])
            width = abs(end_point[0] - start_point[0])
            height = abs(end_point[1] - start_point[1])
            
            # Almacenar región
            regions[region_name] = {
                'left': left,
                'top': top,
                'width': width,
                'height': height
            }
            
            print(f"Región '{region_name}' añadida: left={left}, top={top}, width={width}, height={height}")
            
            # Dibujar rectángulo final
            cv2.rectangle(img_copy, 
                         (left, top), 
                         (left + width, top + height),
                         (0, 255, 0), 2)
            cv2.putText(img_copy, region_name, (left, top - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Avanzar a la siguiente región
            current_region_index += 1
            
            # Reiniciar puntos
            start_point = None
            end_point = None
    
    cv2.imshow('Region Selector', img_copy)

def generate_code():
    """Generar código Python para las regiones seleccionadas."""
    if not regions or "poker_table" not in regions:
        return "No se han seleccionado todas las regiones necesarias."
    
    code = "# Definiciones de regiones para la mesa de póker\n\n"
    
    # Región principal de la mesa
    code += "# Región principal de la mesa de póker\n"
    code += "MANUAL_GAME_REGION = {\n"
    poker_table = regions["poker_table"]
    code += f"    \"top\": {poker_table['top']},\n"
    code += f"    \"left\": {poker_table['left']},\n"
    code += f"    \"width\": {poker_table['width']},\n"
    code += f"    \"height\": {poker_table['height']}\n"
    code += "}\n\n"
    
    # Sub-regiones
    code += "# Sub-regiones dentro de la mesa de póker\n"
    code += "sub_regions = {\n"
    
    for name, coords in regions.items():
        if name != "poker_table":
            code += f"    \"{name}\": {{\n"
            code += f"        \"top\": {coords['top']},\n"
            code += f"        \"left\": {coords['left']},\n"
            code += f"        \"width\": {coords['width']},\n"
            code += f"        \"height\": {coords['height']}\n"
            code += "    },\n"
    
    code += "}\n\n"
    
    # Añadir función para simplificar buttons
    code += "# Función auxiliar para acceder a botones fácilmente\n"
    code += "def get_buttons():\n"
    code += "    return {\n"
    for name, coords in regions.items():
        if "button" in name:
            code += f"        \"{name.replace('_button', '')}\": {{\n"
            code += f"            \"top\": {coords['top']},\n"
            code += f"            \"left\": {coords['left']},\n"
            code += f"            \"width\": {coords['width']},\n"
            code += f"            \"height\": {coords['height']}\n"
            code += "        },\n"
    code += "    }\n"
    
    return code

def verify_selections():
    """Verificar que todas las regiones requeridas están seleccionadas correctamente."""
    missing = [r for r in region_order if r not in regions]
    if missing:
        return False, f"Faltan regiones: {', '.join(missing)}"
    
    # Verificar que la mesa contiene todas las subregiones
    poker_table = regions.get("poker_table", {"top": 0, "left": 0, "width": 0, "height": 0})
    for name, region in regions.items():
        if name != "poker_table":
            if (region["left"] < poker_table["left"] or 
                region["top"] < poker_table["top"] or
                region["left"] + region["width"] > poker_table["left"] + poker_table["width"] or
                region["top"] + region["height"] > poker_table["top"] + poker_table["height"]):
                return False, f"La región {name} está fuera de la mesa de póker!"
    
    return True, "Todas las regiones están correctamente definidas."

def main():
    """Función principal para ejecutar la herramienta de selección de regiones."""
    global image, current_region_index
    
    print("Herramienta avanzada de selección de regiones para Poker AI")
    print("----------------------------------------------------------")
    print("Instrucciones:")
    print("1. Selecciona primero la MESA COMPLETA DE PÓKER")
    print("2. Luego selecciona las demás regiones en el orden indicado")
    print("3. Presiona 'q' al terminar para generar el código")
    print("4. Presiona 'r' para reiniciar si cometes un error")
    
    # Tomar captura de pantalla
    image = take_screenshot()
    
    # Crear ventana y configurar callback del mouse
    cv2.namedWindow('Region Selector')
    cv2.setMouseCallback('Region Selector', mouse_callback)
    
    while True:
        # Si ya seleccionamos todas las regiones, mostramos un mensaje
        if current_region_index >= len(region_order):
            valid, message = verify_selections()
            if valid:
                cv2.putText(image, "Todas las regiones seleccionadas! Presiona 'q' para continuar", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(image, f"Error: {message}", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                current_region_index = region_order.index(message.split(": ")[1].split(",")[0]) if ": " in message else 0
        
        # Hacer una copia de la imagen para dibujar
        img_copy = image.copy()
        
        # Dibujar regiones existentes
        for name, coords in regions.items():
            cv2.rectangle(img_copy, 
                         (coords['left'], coords['top']), 
                         (coords['left'] + coords['width'], coords['top'] + coords['height']),
                         (0, 255, 0), 2)
            cv2.putText(img_copy, name, (coords['left'], coords['top'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar instrucciones actuales
        if current_region_index < len(region_order):
            instruction = f"Selecciona: {region_order[current_region_index]} ({current_region_index+1}/{len(region_order)})"
            cv2.putText(img_copy, instruction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow('Region Selector', img_copy)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Salir con 'q'
        if key == ord('q'):
            break
        
        # Reiniciar con 'r'
        if key == ord('r'):
            regions.clear()
            current_region_index = 0
            print("Todas las regiones borradas. Comenzando de nuevo.")
    
    # Verificar que todas las regiones estén seleccionadas
    valid, message = verify_selections()
    if not valid:
        print(f"ADVERTENCIA: {message}")
        if input("¿Quieres continuar de todos modos? (s/n): ").lower() != 's':
            print("Operación cancelada.")
            cv2.destroyAllWindows()
            return
    
    # Generar y mostrar código
    code = generate_code()
    print("\n----- Código Python Generado -----")
    print(code)
    
    # Guardar código en archivo
    with open("poker_regions.py", "w") as f:
        f.write(code)
    print("\nCódigo guardado en poker_regions.py")
    
    # Guardar captura con regiones marcadas
    result_img = image.copy()
    for name, coords in regions.items():
        color = (0, 255, 0) if name == "poker_table" else (0, 0, 255)
        cv2.rectangle(result_img, 
                     (coords['left'], coords['top']), 
                     (coords['left'] + coords['width'], coords['top'] + coords['height']),
                     color, 2)
        cv2.putText(result_img, name, (coords['left'], coords['top'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Crear directorio debug si no existe
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/poker_regions.png", result_img)
    print("Imagen con regiones marcadas guardada en debug/poker_regions.png")
    
    # Limpiar
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()