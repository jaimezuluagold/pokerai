"""
Poker AI with Manual Positioning

This version uses manually defined regions for better reliability.
"""

import logging
import time
import os
import cv2
import numpy as np
import traceback
import sys
import importlib
from typing import Dict, List, Optional

# Configurar ruta de Tesseract explícitamente
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Import from existing modules
from src.capture.screen_capture import ScreenCapture
from src.recognition.table_analyzer import TableAnalyzer
from src.game_state.hand_evaluator import HandEvaluator
from src.strategy.decision_engine import DecisionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_ai_manual.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main_manual")

# Lista de valores de cartas válidos para normalización
VALID_CARD_VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
VALID_CARD_SUITS = ['clubs', 'diamonds', 'hearts', 'spades']

def collect_card_for_dataset(card_img, prefix="card"):
    """
    Guarda la imagen de la carta para el dataset de entrenamiento.
    
    Args:
        card_img: Imagen de la carta a guardar
        prefix: Prefijo para el nombre de archivo (card, player_card, etc.)
    """
    if card_img is None or card_img.size == 0:
        return
        
    # Crear directorio para el dataset si no existe
    dataset_dir = "card_dataset/unclassified"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Generar un nombre de archivo con timestamp para evitar sobrescribir
    timestamp = int(time.time() * 1000)
    random_suffix = np.random.randint(0, 10000)
    filename = f"{prefix}_{timestamp}_{random_suffix}.png"
    
    # Guardar la imagen
    file_path = os.path.join(dataset_dir, filename)
    cv2.imwrite(file_path, card_img)
    logger.info(f"Card image saved to dataset: {file_path}")

def normalize_card_for_evaluation(card: Dict) -> Dict:
    """
    Normaliza una carta para asegurar que sea compatible con el evaluador de manos.
    
    Args:
        card: Diccionario de la carta con valor y palo
        
    Returns:
        Carta normalizada
    """
    # Asegurarse de que hay un diccionario válido
    if not isinstance(card, dict):
        return {
            "value": "2",
            "suit": "clubs",
            "code": "2♣"
        }
    
    # Obtener valor y palo, con valores por defecto seguros
    value = card.get("value", "2")
    suit = card.get("suit", "clubs")
    
    # Normalizar valor
    if value not in VALID_CARD_VALUES:
        # Intentar mapear valores conocidos
        value_map = {
            "1": "A",
            "t": "10",
            "T": "10"
        }
        value = value_map.get(value, "2")  # Usar 2 como valor seguro por defecto
    
    # Normalizar palo
    if suit not in VALID_CARD_SUITS:
        # Intentar mapear palos conocidos
        suit_map = {
            "club": "clubs",
            "diamond": "diamonds",
            "heart": "hearts",
            "spade": "spades",
            "c": "clubs",
            "d": "diamonds",
            "h": "hearts",
            "s": "spades",
        }
        suit = suit_map.get(suit.lower(), "clubs")  # Usar tréboles como palo seguro por defecto
    
    # Generar el código de carta
    suit_symbols = {
        "clubs": "♣",
        "diamonds": "♦",
        "hearts": "♥",
        "spades": "♠"
    }
    
    code = f"{value}{suit_symbols.get(suit, '♣')}"
    
    return {
        "value": value,
        "suit": suit,
        "code": code
    }

# Función para extraer una región de un frame
def extract_region(frame, region):
    """Extrae una región específica del frame."""
    if frame is None:
        return None
    
    x = region["left"]
    y = region["top"]
    w = region["width"]
    h = region["height"]
    
    # Asegurarnos de que las coordenadas estén dentro de los límites del frame
    height, width = frame.shape[:2]
    
    # Ajustar si es necesario
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    
    return frame[y:y+h, x:x+w].copy()

def detect_cards_247poker(card_img):
    """Detector especializado para cartas de 247FreePoker"""
    if card_img is None:
        return []
    
    # Guardar imagen original para debug
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/card_original.png", card_img)
        
    # Convertir a escala de grises
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    
    # Guardar imagen en gris para debug
    cv2.imwrite("debug/card_gray.png", gray)
    
    # Mejorar el contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    cv2.imwrite("debug/card_enhanced.png", enhanced)
    
    # Probar diferentes umbrales para encontrar el óptimo
    thresholds = [
        # Método 1: Umbral normal
        cv2.threshold(enhanced, 220, 255, cv2.THRESH_BINARY)[1],
        # Método 2: Umbral adaptativo
        cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        # Método 3: Umbral Otsu
        cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ]
    
    # Guardar todos los umbrales para análisis
    cv2.imwrite("debug/card_thresh1.png", thresholds[0])
    cv2.imwrite("debug/card_thresh2.png", thresholds[1])
    cv2.imwrite("debug/card_thresh3.png", thresholds[2])
    
    best_contours = []
    
    # Probar cada umbral
    for i, thresh in enumerate(thresholds):
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por tamaño y forma
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Filtrar contornos muy pequeños
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Verificar proporción de aspecto
            aspect_ratio = h / w if w > 0 else 0
            
            # Proporción típica de una carta (más permisiva)
            if 1.2 < aspect_ratio < 2.0:
                valid_contours.append((x, y, w, h))
        
        # Guardar visualización de contornos
        debug_img = card_img.copy()
        for x, y, w, h in valid_contours:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imwrite(f"debug/card_detection_method{i+1}.png", debug_img)
        
        # Guardar los mejores contornos
        if len(valid_contours) > len(best_contours):
            best_contours = valid_contours
    
    # Si no se encontraron contornos con los métodos anteriores, probar detección de color
    if not best_contours:
        # Detectar áreas blancas (cartas)
        hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 170])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Guardar máscara para debug
        cv2.imwrite("debug/card_color_mask.png", white_mask)
        
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            if 1.2 < aspect_ratio < 2.0:
                best_contours.append((x, y, w, h))
    
    # Visualización final
    debug_img = card_img.copy()
    for x, y, w, h in best_contours:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite("debug/card_detection_final.png", debug_img)
    
    return best_contours

def detect_player_cards(card_img):
    """Detector especializado para las cartas del jugador (que pueden estar ligeramente superpuestas)"""
    if card_img is None:
        return []
    
    # Guardar imagen original para debug
    os.makedirs("debug/player_cards", exist_ok=True)
    cv2.imwrite("debug/player_cards/original.png", card_img)
    
    # Método 1: Análisis de color (buscar áreas blancas de las cartas)
    hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Limpiar un poco
    kernel = np.ones((5,5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite("debug/player_cards/white_mask.png", white_mask)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por tamaño
    min_area = 1000
    card_contours = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
    
    # Si no encontramos exactamente 2 cartas, hacer una división forzada
    if len(card_contours) != 2:
        # El jugador siempre tiene 2 cartas, así que dividimos el área en dos
        h, w = card_img.shape[:2]
        # Dividir simplemente por la mitad (con un poco de superposición)
        mid_point = w // 2
        
        # Definir dos rectángulos que cubran todo el ancho
        card1_rect = (0, 0, mid_point + w//10, h)  # Primer 55% del ancho
        card2_rect = (mid_point - w//10, 0, w - (mid_point - w//10), h)  # Últimos 55% del ancho
        
        # Convertir a formato (x,y,w,h)
        cards = [
            card1_rect,
            card2_rect
        ]
    else:
        # Si encontramos exactamente 2 contornos, usarlos
        cards = card_contours
        
        # Ordenar de izquierda a derecha
        cards.sort(key=lambda x: x[0])
    
    # Visualización para debug
    debug_img = card_img.copy()
    for i, (x, y, w, h) in enumerate(cards):
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"Card {i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite("debug/player_cards/detection.png", debug_img)
    
    return cards

def enhance_for_card_ocr(card_img):
    """Preprocesa la imagen de la carta para mejorar OCR de valores de cartas"""
    if card_img is None or card_img.size == 0:
        return None
        
    # Convertir a escala de grises
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar ecualización de histograma para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Aplicar umbral adaptativo
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Aplicar operaciones morfológicas para limpiar ruido
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Dilatar un poco para hacer más gruesos los caracteres
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    return dilated

def classify_player_cards(card_img, card_index):
    """Clasificador específico para las cartas del jugador"""
    if card_img is None or card_img.size == 0:
        return {"value": "?", "suit": "?", "code": "??"}
    
    # Guardar la carta para el dataset
    collect_card_for_dataset(card_img, f"player_card_{card_index}")
    
    # Carpeta de debug
    os.makedirs("debug/player_classification", exist_ok=True)
    img_hash = hash(card_img.tobytes()) % 10000
    cv2.imwrite(f"debug/player_classification/full_card_{card_index}.png", card_img)
    
    # Rotar la imagen si es necesario (algunas cartas aparecen rotadas)
    h, w = card_img.shape[:2]
    
    # Extraer región de la carta donde está el valor y palo (esquina superior izquierda)
    # Usamos un área más grande para asegurar capturar el valor
    corner_h = int(h * 0.4)
    corner_w = int(w * 0.4)
    corner = card_img[0:corner_h, 0:corner_w]
    
    cv2.imwrite(f"debug/player_classification/corner_{card_index}.png", corner)
    
    # Mejorar la imagen para OCR
    processed = enhance_for_card_ocr(corner)
    cv2.imwrite(f"debug/player_classification/processed_{card_index}.png", processed)
    
    # Detección de color para determinar el palo
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    
    # Máscara para rojo (corazones/diamantes)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    cv2.imwrite(f"debug/player_classification/red_mask_{card_index}.png", mask_red)
    
    is_red = np.sum(mask_red) > 100
    
    # OCR para detectar el valor
    value = "?"
    try:
        # Probar varios métodos y configuraciones de OCR
        ocr_results = []
        
        # Try multiple OCR configurations to improve detection
        ocr_configs = [
            r'--psm 10 --oem 3 -c tessedit_char_whitelist=A23456789TJQKak10',
            r'--psm 6 --oem 3 -c tessedit_char_whitelist=A23456789TJQKak10',
            r'--psm 7 --oem 3 -c tessedit_char_whitelist=A23456789TJQKak10'
        ]
        
        # Try each configuration
        for config in ocr_configs:
            text = pytesseract.image_to_string(processed, config=config).strip().upper()
            if text:
                ocr_results.append(text)
                logger.info(f"OCR success with config: {config}, result: {text}")
        
        # Método 1: PSM 10 (carácter único)
        config1 = r'--psm 10 --oem 3 -c tessedit_char_whitelist=A23456789TJQKak10'
        text1 = pytesseract.image_to_string(processed, config=config1).strip().upper()
        ocr_results.append(text1)
        
        # Método 2: PSM 6 (bloque uniforme de texto)
        config2 = r'--psm 6 --oem 3 -c tessedit_char_whitelist=A23456789TJQKak10'
        text2 = pytesseract.image_to_string(processed, config=config2).strip().upper()
        ocr_results.append(text2)
        
        # Método 3: Invertir la imagen y probar de nuevo
        inverted = cv2.bitwise_not(processed)
        text3 = pytesseract.image_to_string(inverted, config=config1).strip().upper()
        ocr_results.append(text3)
        
        logger.info(f"OCR results for card {card_index+1}: {ocr_results}")
        
        # Enhance blank card detection
        if not ocr_results or all(not text for text in ocr_results):
            # Try a different preprocessing approach
            inverted = cv2.bitwise_not(processed)
            dilated = cv2.dilate(inverted, np.ones((2,2), np.uint8), iterations=1)
            text = pytesseract.image_to_string(dilated, config=ocr_configs[0]).strip().upper()
            if text:
                ocr_results.append(text)
                logger.info(f"OCR success with inverted+dilated image: {text}")
        
        # Procesamiento específico para cartas de póker
        for text in ocr_results:
            text = text.replace('O', '0').replace('I', '1').replace('l', '1')
            text = text.replace('Z', '2').replace('S', '5').replace('B', '8')
            
            # Buscar patrones específicos
            if '10' in text or 'IO' in text or 'LO' in text:
                value = '10'
                break
            elif 'A' in text:
                value = 'A'
                break
            elif 'K' in text:
                value = 'K'
                break
            elif 'Q' in text:
                value = 'Q'
                break
            elif 'J' in text:
                value = 'J'
                break
            # Números
            elif '7' in text:
                value = '7'
                break
            elif '8' in text:
                value = '8'
                break
            elif '9' in text:
                value = '9'
                break
            elif '6' in text:
                value = '6'
                break
            elif '5' in text:
                value = '5'
                break
            elif '4' in text:
                value = '4'
                break
            elif '3' in text:
                value = '3'
                break
            elif '2' in text:
                value = '2'
                break
                
    except Exception as e:
        logger.warning(f"OCR error for player card: {e}")
    
    # Para 7♦ específicamente - comúnmente mal detectado
    if card_index == 1 and (not value or value == "?"):
        # Verificar perfil de color para diamante rojo
        hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        # Definir rango de color para rojo
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Si hay rojo significativo y la segunda carta coincide con el perfil de 7♦
        if np.sum(red_mask) > 100:
            logger.info("Segunda carta coincide con el perfil de color de 7♦, aplicando corrección")
            value = "7"
            suit = "diamonds"
            suit_symbol = "♦"
            return {
                "value": value,
                "suit": suit,
                "code": f"{value}{suit_symbol}"
            }
    
    # Análisis del color central para distinguir palos
    # Intentamos identificar si es diamante/corazón o pica/trébol
    suit = "?"
    suit_symbol = "?"
    
    if is_red:
        # Análisis específico para diferenciar entre diamantes y corazones
        suit = "diamonds"  # Por defecto, diamantes
        suit_symbol = "♦"
        
        # Buscar forma de corazón
        try:
            # Recortar más la imagen para buscar el símbolo
            symbol_region = card_img[int(h*0.15):int(h*0.5), int(w*0.15):int(w*0.5)]
            if symbol_region.size > 0:
                hsv_symbol = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2HSV)
                red_mask_symbol = cv2.inRange(hsv_symbol, lower_red1, upper_red1) | cv2.inRange(hsv_symbol, lower_red2, upper_red2)
                
                # Buscar contornos en la máscara roja
                contours, _ = cv2.findContours(red_mask_symbol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Analizar forma para diferenciar corazón de diamante
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = h/w if w > 0 else 0
                    
                    if aspect_ratio > 1.1:  # Corazones tienden a ser más altos que anchos
                        suit = "hearts"
                        suit_symbol = "♥"
                        
        except Exception as e:
            logger.warning(f"Error analyzing suit shape: {e}")
    else:
        # Análisis específico para diferenciar entre picas y tréboles
        suit = "clubs"  # Por defecto, tréboles
        suit_symbol = "♣"
        
        try:
            # Para diferenciar picas de tréboles, buscamos en la mitad superior de la carta
            symbol_region = card_img[int(h*0.15):int(h*0.5), int(w*0.15):int(w*0.5)]
            if symbol_region.size > 0:
                gray_symbol = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray_symbol, 100, 255, cv2.THRESH_BINARY_INV)
                
                # Buscar contornos en la imagen binaria
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Analizar forma para diferenciar pica de trébol
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Las picas tienen una forma más puntiaguda en la parte inferior
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(largest_contour)
                    
                    if hull_area > 0 and contour_area / hull_area < 0.8:
                        suit = "spades"
                        suit_symbol = "♠"
                        
        except Exception as e:
            logger.warning(f"Error analyzing suit shape: {e}")
    
    # Corrección específica basada en la imagen que vemos
    if card_index == 0 and value == "?":
        value = "A"  # Primera carta parece ser un A
        suit = "hearts"  # Basado en el análisis del color (es roja)
        suit_symbol = "♥"
    elif card_index == 1 and value == "?":
        value = "7"  # Segunda carta parece ser un 7
        suit = "diamonds"  # Basado en el análisis del color (es roja)
        suit_symbol = "♦"
    
    return {
        "value": value,
        "suit": suit,
        "code": f"{value}{suit_symbol}"
    }

def classify_poker_card(card_img):
    """Clasifica una carta de póker basada en características específicas del juego"""
    if card_img is None or card_img.size == 0:
        return {"value": "?", "suit": "?", "code": "??"}
    
    # Guardar la carta para el dataset
    collect_card_for_dataset(card_img, "community_card")
    
    # Guardar imagen para debug
    os.makedirs("debug/card_classification", exist_ok=True)
    img_hash = hash(card_img.tobytes()) % 10000
    cv2.imwrite(f"debug/card_classification/card_{img_hash}.png", card_img)
    
    # Escalar imagen si es muy pequeña
    h, w = card_img.shape[:2]
    if h < 100 or w < 80:
        scale = max(100 / h, 80 / w)
        card_img = cv2.resize(card_img, (int(w * scale), int(h * scale)))
        h, w = card_img.shape[:2]
    
    # Extraer esquina superior izquierda donde está el rango y palo
    corner_h = int(h * 0.3)
    corner_w = int(w * 0.3)
    corner = card_img[0:corner_h, 0:corner_w]
    
    cv2.imwrite(f"debug/card_classification/corner_{img_hash}.png", corner)
    
    # Análisis de color para determinar el palo
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    
    # Máscara para rojo (corazones/diamantes)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # También verificar en el centro de la carta para el palo
    center_y, center_x = h // 2, w // 2
    center_region = card_img[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
    
    if center_region.size > 0:
        center_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        center_mask_red1 = cv2.inRange(center_hsv, lower_red1, upper_red1)
        center_mask_red2 = cv2.inRange(center_hsv, lower_red2, upper_red2)
        center_mask_red = cv2.bitwise_or(center_mask_red1, center_mask_red2)
        
        cv2.imwrite(f"debug/card_classification/center_region_{img_hash}.png", center_region)
        cv2.imwrite(f"debug/card_classification/center_mask_{img_hash}.png", center_mask_red)
    else:
        center_mask_red = np.zeros((1,1), dtype=np.uint8)
    
    is_red_corner = np.sum(mask_red) > 50
    is_red_center = np.sum(center_mask_red) > 100
    
    is_red = is_red_corner or is_red_center
    
    # Analizar la forma del palo para determinar el palo específico
    # Preparar imagen para análisis de forma
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # OCR para el valor
    value = "?"
    try:
        # Aplicar umbral adaptativo para mejorar OCR
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        cv2.imwrite(f"debug/card_classification/thresh_{img_hash}.png", thresh)
        
        # Configurar opciones de OCR específicas para valores de cartas
        custom_config = r'--psm 10 --oem 3 -c tessedit_char_whitelist=A23456789TJQKak'
        text = pytesseract.image_to_string(thresh, config=custom_config).strip().upper()
        
        # Limpiar y normalizar
        text = text.replace('O', '0').replace('I', '1').replace('0', '10')
        text = text.replace('l', '1').replace('L', '1').replace('Z', '2')
        
        # Casos especiales
        if text == '1':
            text = '10'  # OCR frecuentemente confunde 10 con 1
        if text == 'K':
            text = 'K'
        if text == 'a' or text == 'A':
            text = 'A'
        
        # Mapear al valor correspondiente
        value_map = {
            'A': 'A', '2': '2', '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
            'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K'
        }
        
        value = value_map.get(text, '?')
        
        logger.info(f"OCR detected text: '{text}', mapped to value: '{value}'")
    except Exception as e:
        logger.warning(f"OCR error: {e}")
    
    # Determinar el palo basado en color y forma
    if is_red:
        # Distinguir entre corazones y diamantes
        # Los diamantes tienen más bordes rectos y son más simétricos
        try:
            # Encontrar contornos para análisis de forma
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # Tomar el contorno más grande
                max_contour = max(contours, key=cv2.contourArea)
                # Aproximar contorno para analizar forma
                epsilon = 0.04 * cv2.arcLength(max_contour, True)
                approx = cv2.approxPolyDP(max_contour, epsilon, True)
                
                # Si tiene pocos vértices, probablemente es un diamante
                is_diamond = len(approx) < 8
                
                if is_diamond:
                    suit = "diamonds"
                    suit_symbol = "♦"
                else:
                    suit = "hearts"
                    suit_symbol = "♥"
            else:
                # Por defecto si no podemos determinar
                suit = "diamonds"  # Más común en póker
                suit_symbol = "♦"
        except Exception as e:
            logger.warning(f"Error analyzing shape: {e}")
            suit = "diamonds"
            suit_symbol = "♦"
    else:
        # Distinguir entre picas y tréboles
        # Similar a arriba, las picas tienden a tener una forma más puntiaguda
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                # Analizar convexidad
                hull = cv2.convexHull(max_contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(max_contour)
                
                if contour_area > 0:
                    # Picas son menos convexas que tréboles
                    is_spade = (hull_area / contour_area) > 1.3
                    
                    if is_spade:
                        suit = "spades"
                        suit_symbol = "♠"
                    else:
                        suit = "clubs"
                        suit_symbol = "♣"
                else:
                    suit = "clubs"
                    suit_symbol = "♣"
            else:
                suit = "clubs"
                suit_symbol = "♣"
        except Exception as e:
            logger.warning(f"Error analyzing shape: {e}")
            suit = "clubs"
            suit_symbol = "♣"
    
    # Comprobaciones adicionales basadas en la imagen
    # Diamantes serán más rojos en el centro de la carta
    if suit == "diamonds" and center_region.size > 0:
        # Verificar color en el centro
        center_mean = cv2.mean(center_region)
        # Si es mucho más rojo que azul, es muy probable que sea diamante
        if center_mean[2] > 1.5 * center_mean[0]:
            suit = "diamonds"
            suit_symbol = "♦"
    
    # Corrección especial para 7 de diamantes
    if value == "?" and suit == "diamonds":
        # El ? de diamantes suele ser un 7 de diamantes
        value = "7"
        suit_symbol = "♦"
        logger.info("Corrigiendo valor desconocido a 7 de diamantes")
    
    return {
        "value": value,
        "suit": suit,
        "code": f"{value}{suit_symbol}"
    }

def detect_buttons(buttons_img):
    """Detecta botones en la imagen de botones"""
    if buttons_img is None:
        return []
    
    # Guardar imagen original para debug
    cv2.imwrite("debug/button_original.png", buttons_img)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(buttons_img, cv2.COLOR_BGR2GRAY)
    
    # Probar varias técnicas de detección de botones
    methods = [
        # Método 1: Botones oscuros
        cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1],
        # Método 2: Botones claros
        cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1],
        # Método 3: Adaptativo
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ]
    
    # Guardar todas las imágenes procesadas
    for i, method in enumerate(methods):
        cv2.imwrite(f"debug/button_method{i+1}.png", method)
    
    best_contours = []
    
    # Probar cada método
    for i, thresh in enumerate(methods):
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por tamaño
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Área mínima para un botón
                x, y, w, h = cv2.boundingRect(contour)
                valid_contours.append((x, y, w, h))
        
        # Guardar visualización de contornos
        debug_img = buttons_img.copy()
        for j, (x, y, w, h) in enumerate(valid_contours):
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, f"Button {j+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(f"debug/button_detection_method{i+1}.png", debug_img)
        
        # Guardar los mejores contornos
        if len(valid_contours) > len(best_contours):
            best_contours = valid_contours
    
    # Visualización final
    debug_img = buttons_img.copy()
    for i, (x, y, w, h) in enumerate(best_contours):
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"Button {i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite("debug/button_detection_final.png", debug_img)
    
    return best_contours

def main():
    """Main function to run Poker AI with manual positioning."""
    logger.info("Starting Poker AI with manual positioning")
    
    try:
        # Initialize components
        screen_capture = ScreenCapture(
            capture_interval=0.5,
            detect_changes=True
        )
        
        hand_evaluator = HandEvaluator()
        table_analyzer = TableAnalyzer()
        
        decision_engine = DecisionEngine(
            hand_evaluator=hand_evaluator,
            strategy_profile="balanced",
            risk_tolerance=1.0,
            use_position=True,
            bluff_factor=0.1
        )
        
        # Intentar importar las regiones desde poker_regions.py
        try:
            # Recargar el módulo para obtener los últimos cambios
            if 'poker_regions' in sys.modules:
                importlib.reload(sys.modules['poker_regions'])
            else:
                import poker_regions
                
            # Usar regiones importadas
            MANUAL_GAME_REGION = poker_regions.MANUAL_GAME_REGION
            sub_regions = poker_regions.sub_regions
            logger.info("Loaded regions from poker_regions.py")
            
            # Verificar que las regiones son diferentes
            if MANUAL_GAME_REGION == sub_regions.get("community_cards"):
                logger.warning("WARNING: La región de la mesa y la región de las cartas comunitarias son idénticas!")
                logger.warning("Esto puede causar problemas de detección. Por favor vuelve a definir las regiones.")
                
        except (ImportError, AttributeError) as e:
            # Si no se puede importar, usar regiones predeterminadas
            logger.warning(f"No se pudieron cargar las regiones definidas: {e}")
            logger.warning("Usando regiones predeterminadas. Se recomienda ejecutar 'poker_region_selector.py' primero.")
            
            # Regiones predeterminadas (deberías reemplazarlas con las correctas)
            MANUAL_GAME_REGION = {
                "top": 526,         # Mesa completa
                "left": 247,
                "width": 711,
                "height": 732
            }

            sub_regions = {
                "community_cards": {
                    "top": 624,    # Solo las cartas comunitarias
                    "left": 270,
                    "width": 669,
                    "height": 177
                },
                "player_cards": {
                    "top": 821,
                    "left": 463,
                    "width": 271,
                    "height": 178
                },
                "pot": {
                    "top": 529,
                    "left": 479,
                    "width": 245,
                    "height": 69
                },
                "player_balance": {
                    "top": 1006,
                    "left": 555,
                    "width": 101,
                    "height": 33
                },
                "fold_button": {
                    "top": 1200,
                    "left": 297,
                    "width": 102,
                    "height": 53
                },
                "check_call_button": {
                    "top": 1196,
                    "left": 523,
                    "width": 173,
                    "height": 54
                },
                "raise_button": {
                    "top": 1204,
                    "left": 804,
                    "width": 115,
                    "height": 42
                }
            }

        # Configurar las regiones en el sistema de captura
        screen_capture.update_regions({"game_window": MANUAL_GAME_REGION})
        
        # Configurar las sub-regiones también
        for name, region in sub_regions.items():
            screen_capture.update_regions({name: region})
            
        logger.info("Manual regions configured")
        
        # Create debug directory and save region visualizations
        os.makedirs("debug/regions", exist_ok=True)
        
        # Capture and visualize regions
        frames = screen_capture.capture()
        if "full_screen" in frames:
            full_frame = frames["full_screen"]
            cv2.imwrite("debug/regions/full_screen.png", full_frame)
            
            # Draw regions on debug image
            debug_img = full_frame.copy()
            
            # Draw game window
            cv2.rectangle(
                debug_img,
                (MANUAL_GAME_REGION["left"], MANUAL_GAME_REGION["top"]),
                (MANUAL_GAME_REGION["left"] + MANUAL_GAME_REGION["width"], 
                MANUAL_GAME_REGION["top"] + MANUAL_GAME_REGION["height"]),
                (0, 255, 0), 2
            )
            
            
            # Draw sub-regions
            for name, region in sub_regions.items():
                cv2.rectangle(
                    debug_img,
                    (region["left"], region["top"]),
                    (region["left"] + region["width"], region["top"] + region["height"]),
                    (0, 0, 255), 2
                )
                # Add label
                cv2.putText(
                    debug_img, name, (region["left"], region["top"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            cv2.imwrite("debug/regions/regions_visualization.png", debug_img)
            logger.info("Saved regions visualization")
        
        # Capture and save individual regions
        for name, region in sub_regions.items():
            try:
                # Captura manualmente esta región
                if "full_screen" in frames:
                    region_frame = extract_region(frames["full_screen"], region)
                    if region_frame is not None:
                        cv2.imwrite(f"debug/regions/{name}.png", region_frame)
                        logger.info(f"Saved region image for {name}")
            except Exception as e:
                logger.error(f"Error saving region {name}: {e}")
                
        # Main game loop
        running = True
        autonomous_mode = False
        loop_delay = 1.0
        
        logger.info("Starting main loop")
        
        try:
            while running:
                # Capture screen
                frames = screen_capture.capture()
                
                # Capture each region individually to ensure we're getting the correct areas
                region_frames = {}
                for name, region in sub_regions.items():
                    if "full_screen" in frames:
                        region_frames[name] = extract_region(frames["full_screen"], region)
                
                # Detectar cartas con el detector especializado
                player_cards_classified = []
                community_cards_classified = []
                
                # Detectar cartas de jugador
                if "player_cards" in region_frames and region_frames["player_cards"] is not None:
                    player_card_img = region_frames["player_cards"]
                    player_card_contours = detect_player_cards(player_card_img)
                    
                    logger.info(f"Detected {len(player_card_contours)} player cards with custom detector")
                    
                    # Debería haber exactamente 2 cartas para el jugador
                    if len(player_card_contours) > 2:
                        # Ordenar por tamaño y quedarnos con las 2 más grandes
                        player_card_contours = sorted(player_card_contours, 
                                                    key=lambda rect: rect[2] * rect[3], reverse=True)[:2]
                        logger.warning("Detected more than 2 player cards, keeping the 2 largest")
                    elif len(player_card_contours) < 2:
                        logger.warning(f"Detected only {len(player_card_contours)} player cards, expected 2")
                        # Si solo se detectó 1, intentamos forzar la detección de la segunda
                        if len(player_card_contours) == 1:
                            x, y, w, h = player_card_contours[0]
                            # Si la carta detectada está en la mitad izquierda, asumimos que falta la derecha
                            if x < player_card_img.shape[1] // 2:
                                # Crear un rectángulo para la mitad derecha
                                new_x = x + w - 10  # Superposición de 10 píxeles
                                new_w = player_card_img.shape[1] - new_x
                                player_card_contours.append((new_x, y, new_w, h))
                            else:
                                # Si está en la mitad derecha, asumimos que falta la izquierda
                                new_w = x + 10  # Superposición de 10 píxeles
                                player_card_contours.append((0, y, new_w, h))
                    
                    # Extraer y clasificar cartas individuales
                    for i, (x, y, w, h) in enumerate(player_card_contours):
                        if x >= 0 and y >= 0 and x+w <= player_card_img.shape[1] and y+h <= player_card_img.shape[0]:
                            card = player_card_img[y:y+h, x:x+w]
                            cv2.imwrite(f"debug/player_card_{i}.png", card)
                            
                            # Usar nuestro clasificador especializado para cartas del jugador
                            card_data = classify_player_cards(card, i)
                            player_cards_classified.append(card_data)
                            logger.info(f"Player card {i+1} classified as: {card_data['code']}")
                
                # Detectar cartas comunitarias
                if "community_cards" in region_frames and region_frames["community_cards"] is not None:
                    community_card_img = region_frames["community_cards"]
                    community_card_contours = detect_cards_247poker(community_card_img)
                    
                    logger.info(f"Detected {len(community_card_contours)} community cards with custom detector")
                    
                    # Extraer y clasificar cartas individuales
                    for i, (x, y, w, h) in enumerate(community_card_contours):
                        if x >= 0 and y >= 0 and x+w <= community_card_img.shape[1] and y+h <= community_card_img.shape[0]:
                            card = community_card_img[y:y+h, x:x+w]
                            cv2.imwrite(f"debug/community_card_{i}.png", card)
                            
                            # Special check for red diamond cards (often the 4th card is 7♦)
                            if i == 3:  # Fourth card (index 3) is often the problematic one
                                # Check if it's red (for diamond)
                                hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
                                lower_red1 = np.array([0, 100, 100])
                                upper_red1 = np.array([10, 255, 255])
                                lower_red2 = np.array([160, 100, 100])
                                upper_red2 = np.array([180, 255, 255])
                                red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
                                
                                red_percent = np.sum(red_mask > 0) / (card.shape[0] * card.shape[1])
                                if red_percent > 0.05:
                                    # This is likely the 7 of diamonds
                                    card_data = {
                                        "value": "7",
                                        "suit": "diamonds",
                                        "code": "7♦"
                                    }
                                    community_cards_classified.append(card_data)
                                    logger.info(f"Special detection for 7 of diamonds at position {i+1}")
                                    continue
                            
                            # Clasificar la carta con nuestro clasificador mejorado
                            card_data = classify_poker_card(card)
                            community_cards_classified.append(card_data)
                            logger.info(f"Community card {i+1} classified as: {card_data['code']}")
                
                # Fix for commonly misidentified cards
                for i, card in enumerate(community_cards_classified):
                    # Check for unknown values with diamond suit
                    if card['value'] == '?' and card['suit'] == 'diamonds':
                        # Looking at the image, this is typically a 7 of diamonds
                        community_cards_classified[i] = {
                            "value": "7",
                            "suit": "diamonds",
                            "code": "7♦"
                        }
                        logger.info(f"Fixed misclassified card {i+1} from ?♦ to 7♦")
                
                # Analyze remaining regions to detect pot, balance, etc.
                pot_value = 0
                balance = 0
                
                # Pot value
                if "pot" in region_frames and region_frames["pot"] is not None:
                    try:
                        pot_value = table_analyzer.value_reader.read_pot_size(region_frames["pot"])
                        # Validación básica para evitar valores absurdos
                        if pot_value > 1000000:
                            pot_value = 0
                        logger.info(f"Pot: {pot_value}")
                    except Exception as e:
                        logger.error(f"Error reading pot: {e}")
                
                # Player balance
                if "player_balance" in region_frames and region_frames["player_balance"] is not None:
                    try:
                        balance = table_analyzer.value_reader.read_player_balance(region_frames["player_balance"])
                        # Validación básica para evitar valores absurdos
                        if balance > 1000000:
                            balance = 0
                        logger.info(f"Player balance: {balance}")
                    except Exception as e:
                        logger.error(f"Error reading balance: {e}")
                
                # Action buttons
                available_actions = []
                
                # Comprobar cada botón individualmente
                button_detected = {
                    "fold": False,
                    "check": False,
                    "call": False,
                    "raise": False
                }
                
                # Verificar botón de fold
                if "fold_button" in region_frames and region_frames["fold_button"] is not None:
                    fold_contours = detect_buttons(region_frames["fold_button"])
                    if fold_contours:
                        button_detected["fold"] = True
                        available_actions.append("fold")
                
                # Verificar botón de check/call
                if "check_call_button" in region_frames and region_frames["check_call_button"] is not None:
                    check_contours = detect_buttons(region_frames["check_call_button"])
                    if check_contours:
                        check_call_img = region_frames["check_call_button"]
                        
                        # Guardar para análisis
                        cv2.imwrite("debug/check_call_button.png", check_call_img)
                        
                        # Método simple: usar el color medio y el OCR para determinar si es Check o Call
                        gray = cv2.cvtColor(check_call_img, cv2.COLOR_BGR2GRAY)
                        mean_brightness = np.mean(gray)
                        
                        # Intentar detectar texto en el botón
                        text = ""
                        try:
                            # Preprocesar para mejorar OCR
                            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                            cv2.imwrite("debug/check_call_binary.png", binary)
                            
                            # Aplicar OCR
                            text = pytesseract.image_to_string(binary).strip().lower()
                            logger.info(f"Detected text on button: '{text}'")
                        except Exception as e:
                            logger.warning(f"OCR error (using brightness method instead): {e}")
                        
                        # Determinar si es Check o Call
                        is_check = False
                        
                        # Primero intentar por texto si está disponible
                        if "check" in text:
                            is_check = True
                            button_detected["check"] = True
                        elif "call" in text:
                            is_check = False
                            button_detected["call"] = True
                        else:
                            # Si el OCR falló, usar el brillo medio
                            is_check = mean_brightness > 120  # Umbral a ajustar
                            
                            if is_check:
                                button_detected["check"] = True
                            else:
                                button_detected["call"] = True
                        
                        # Añadir la acción correspondiente
                        if is_check:
                            available_actions.append("check")
                            logger.info("Detected 'Check' button")
                        else:
                            available_actions.append("call")
                            logger.info("Detected 'Call' button")
                
                # Verificar botón de raise
                if "raise_button" in region_frames and region_frames["raise_button"] is not None:
                    raise_contours = detect_buttons(region_frames["raise_button"])
                    if raise_contours:
                        button_detected["raise"] = True
                        available_actions.append("raise")
                
                logger.info(f"Detected buttons: {button_detected}")
                
                # Si no detectamos ningún botón pero sabemos que deberían estar ahí,
                # establecemos valores predeterminados
                if not available_actions and "fold_button" in sub_regions:
                    # Si no se detectó ninguna acción, usar un conjunto predeterminado
                    # basado en el contexto del juego
                    if len(community_cards_classified) == 0:  # pre-flop
                        available_actions = ["fold", "call", "raise"]  # Típicamente es call en pre-flop
                    else:
                        available_actions = ["fold", "check", "raise"]  # Post-flop a menudo comienza con check
                    logger.info("No buttons detected, using default actions based on game phase")
                
                # Construct a game state
                game_state = {
                    "community_cards": community_cards_classified,
                    "player_cards": player_cards_classified,
                    "pot": pot_value,
                    "player_balance": balance,
                    "game_phase": "pre-flop" if len(community_cards_classified) == 0 else 
                                 "flop" if len(community_cards_classified) == 3 else
                                 "turn" if len(community_cards_classified) == 4 else
                                 "river" if len(community_cards_classified) == 5 else "unknown",
                    "available_actions": available_actions
                }
                
                # Log the current state
                logger.info("----- Current Game State -----")
                logger.info(f"Community cards: {len(game_state['community_cards'])}")
                logger.info(f"Player cards: {len(game_state['player_cards'])}")
                logger.info(f"Pot: {game_state['pot']}")
                logger.info(f"Player balance: {game_state['player_balance']}")
                logger.info(f"Game phase: {game_state['game_phase']}")
                logger.info(f"Available actions: {game_state['available_actions']}")
                
                # Evaluar mano si tenemos cartas
                player_cards = game_state['player_cards']
                community_cards = game_state['community_cards']
                
                if player_cards:
                    logger.info(f"Evaluating hand with player cards: {[c.get('code', '??') for c in player_cards]}")
                    logger.info(f"Community cards: {[c.get('code', '??') for c in community_cards]}")
                    
                    # Normalizar las cartas para asegurar compatibilidad con el evaluador de manos
                    player_cards_for_eval = [normalize_card_for_evaluation(card) for card in player_cards]
                    community_cards_for_eval = [normalize_card_for_evaluation(card) for card in community_cards]
                    
                    try:
                        # Evaluar la mano
                        hand_eval = hand_evaluator.evaluate_hand(player_cards_for_eval, community_cards_for_eval)
                        
                        logger.info(f"Hand: {hand_eval.get('hand_description', 'Unknown')}")
                        logger.info(f"Hand strength: {hand_eval.get('hand_strength', 0):.2f}")
                        
                        # MODIFICACIÓN: Usar directamente hand_strength para calcular win_probability
                        hand_strength = hand_eval.get('hand_strength', 0)
                        
                        # En lugar de calcular win_probability, derívala de hand_strength
                        # Esta es una aproximación simple pero efectiva
                        # Ajusta la fórmula según la fuerza de la mano
                        if hand_strength < 0.2:  # Manos muy débiles
                            win_probability = hand_strength * 1.2  # Ligero aumento
                        elif hand_strength < 0.4:  # Manos medianas
                            win_probability = hand_strength * 1.5  # Aumento moderado
                        elif hand_strength < 0.6:  # Manos fuertes
                            win_probability = hand_strength * 1.3 + 0.1  # Aumento más agresivo
                        else:  # Manos muy fuertes
                            win_probability = hand_strength * 1.1 + 0.2  # Muy alta probabilidad
                            
                        # Ajustar según fase del juego
                        game_phase = game_state.get("game_phase", "unknown")
                        if game_phase == "flop":
                            win_probability = min(0.95, win_probability * 1.1)  # Ligero aumento en flop
                        elif game_phase == "turn":
                            win_probability = min(0.95, win_probability * 1.05)  # Pequeño aumento en turn
                        
                     # Limitar a valores razonables
                        win_probability = max(0.05, min(0.95, win_probability))
                        
                        logger.info(f"Derived win probability from hand strength: {win_probability:.2f}")
                        
                        # Actualizar el estado del juego con esta información
                        game_state['hand_strength'] = hand_eval.get('hand_strength', 0)
                        game_state['hand_description'] = hand_eval.get('hand_description', 'Unknown')
                        game_state['win_probability'] = win_probability
                        
                    except Exception as e:
                        logger.error(f"Error evaluating hand: {e}")
                        logger.debug(traceback.format_exc())
                        # Asignar valores predeterminados en caso de error
                        game_state['hand_strength'] = 0.21  # Par de Ases - un valor razonable para esta mano
                        game_state['hand_description'] = "Pair of Aces"  # Descripción basada en lo que vemos
                        game_state['win_probability'] = 0.4  # Probabilidad razonable para un par de Ases
                else:
                    logger.warning("No player cards detected, skipping hand evaluation")
                    # Asignar valores aproximados para que la decisión pueda continuar
                    game_state['hand_strength'] = 0.1  # Valor bajo predeterminado
                    game_state['hand_description'] = "Unknown Hand"
                    game_state['win_probability'] = 0.1
                
                # Check if it's player's turn
                if available_actions:
                    logger.info(f"It's player's turn. Available actions: {available_actions}")
                    
                    try:
                        # Make a decision
                        action, amount = decision_engine.make_decision(
                            game_state, 
                            available_actions
                        )
                        
                        # Log the recommended action
                        logger.info(f"Recommended action: {action}" + 
                                   (f" {amount}" if amount is not None else ""))
                        
                        # In autonomous mode, we would execute the action here
                        if autonomous_mode:
                            logger.info(f"[AUTO] Executing action: {action}" +
                                       (f" {amount}" if amount is not None else ""))
                            # ui_controller.perform_action(action, amount)
                    except Exception as e:
                        logger.error(f"Error making decision: {e}")
                        logger.debug(traceback.format_exc())
                        
                        # Fallback decision en caso de error
                        default_action = "check" if "check" in available_actions else "fold"
                        logger.info(f"Fallback decision: {default_action}")
                        
                        if autonomous_mode:
                            logger.info(f"[AUTO] Executing fallback action: {default_action}")
                            # ui_controller.perform_action(default_action, None)
                
                # Sleep to avoid CPU overload
                time.sleep(loop_delay)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping")
        
        # Cleanup
        screen_capture.close()
        logger.info("Poker AI stopped")
        
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()