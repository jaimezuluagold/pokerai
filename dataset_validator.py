import os
import sys
import yaml

class PokerDatasetValidator:
    def __init__(self, base_dir):
        """
        Inicializa el validador de dataset de póker
        
        :param base_dir: Directorio base del dataset
        """
        self.base_dir = base_dir
        self.splits = ['train', 'test', 'val']
        
        # Verificar si el directorio base existe
        if not os.path.exists(base_dir):
            print(f"❌ ERROR: El directorio {base_dir} no existe!")
            sys.exit(1)
    
    def validate_dataset(self):
        """
        Valida la estructura completa del dataset
        """
        print("\n--- Validación Detallada de Dataset ---")
        print(f"Directorio base: {self.base_dir}")
        
        # Validación de estructura de directorios
        directory_check = self._validate_directory_structure()
        
        # Si la estructura de directorios no es válida, salir
        if not directory_check:
            print("❌ La estructura de directorios no es válida. Por favor, revisa tu configuración.")
            return False
        
        # Validación de consistencia de imágenes y etiquetas
        consistency_results = []
        for split in self.splits:
            print(f"\n--- Validando {split.upper()} ---")
            result = self._validate_split_consistency(split)
            consistency_results.append(result)
        
        # Generar informe resumen
        self._generate_dataset_summary()
        
        # Verificar si todos los splits pasaron la validación
        return all(consistency_results)
    
    def _validate_directory_structure(self):
        """
        Verifica la existencia de directorios esperados
        
        :return: True si la estructura es válida, False en caso contrario
        """
        print("\nValidando Estructura de Directorios:")
        
        all_valid = True
        
        # Verificar existencia de splits
        for split in self.splits:
            split_dir = os.path.join(self.base_dir, split)
            
            # Verificar existencia del split
            if not os.path.exists(split_dir):
                print(f"❌ Falta directorio: {split}")
                all_valid = False
                continue
            
            # Verificar subdirectorios de cada split
            required_subdirs = ['images', 'labels']
            for subdir in required_subdirs:
                subdir_path = os.path.join(split_dir, subdir)
                if not os.path.exists(subdir_path):
                    print(f"❌ Falta subdirectorio: {subdir_path}")
                    all_valid = False
                else:
                    print(f"✅ Directorio {subdir_path} existe")
        
        return all_valid
    
    def _validate_split_consistency(self, split):
        """
        Valida la consistencia de imágenes y etiquetas para un split específico
        
        :param split: Nombre del split (train, test, val)
        :return: True si es consistente, False en caso contrario
        """
        # Rutas de imágenes y labels
        images_dir = os.path.join(self.base_dir, split, 'images')
        labels_dir = os.path.join(self.base_dir, split, 'labels')
        
        # Verificar que los directorios existan
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"❌ Directorios de {split} no encontrados")
            return False
        
        # Obtener nombres de archivos sin extensión
        try:
            image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))}
            label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) 
                           if f.lower().endswith('.txt')}
        except Exception as e:
            print(f"Error al leer directorios: {e}")
            return False
        
        # Verificar consistencia
        missing_labels = image_files - label_files
        missing_images = label_files - image_files
        
        print(f"Total de imágenes: {len(image_files)}")
        print(f"Total de etiquetas: {len(label_files)}")
        
        is_consistent = True
        
        if missing_labels:
            print(f"⚠️ Imágenes sin etiquetas ({len(missing_labels)}):")
            print(list(missing_labels)[:10])  # Mostrar primeros 10
            is_consistent = False
        else:
            print("✅ Todas las imágenes tienen etiquetas")
        
        if missing_images:
            print(f"⚠️ Etiquetas sin imágenes ({len(missing_images)}):")
            print(list(missing_images)[:10])  # Mostrar primeros 10
            is_consistent = False
        else:
            print("✅ Todas las etiquetas tienen imágenes correspondientes")
        
        return is_consistent
    
    def _generate_dataset_summary(self):
        """
        Genera un resumen del dataset para YOLO
        """
        # Contar clases únicas en las etiquetas
        unique_classes = set()
        for split in self.splits:
            labels_dir = os.path.join(self.base_dir, split, 'labels')
            
            # Verificar que el directorio de etiquetas exista
            if not os.path.exists(labels_dir):
                print(f"❌ Directorio de etiquetas no encontrado: {labels_dir}")
                continue
            
            # Revisar todas las etiquetas para extraer clases únicas
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    try:
                        with open(os.path.join(labels_dir, label_file), 'r') as f:
                            unique_classes.update([line.split()[0] for line in f])
                    except Exception as e:
                        print(f"Error al leer archivo de etiquetas {label_file}: {e}")
        
        # Generar archivo data.yaml
        data_config = {
            'train': os.path.join(self.base_dir, 'train', 'images'),
            'val': os.path.join(self.base_dir, 'val', 'images'),
            'test': os.path.join(self.base_dir, 'test', 'images'),
            'nc': len(unique_classes),
            'names': list(unique_classes)
        }
        
        yaml_path = os.path.join(self.base_dir, 'data.yaml')
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            print("\n--- Resumen del Dataset ---")
            print(f"Clases únicas detectadas: {len(unique_classes)}")
            print(f"Archivo data.yaml generado en: {yaml_path}")
        except Exception as e:
            print(f"❌ Error al generar data.yaml: {e}")

def main():
    # Ruta específica proporcionada
    DATASET_DIR = r"C:\Users\jaime_5dwv2oh\poker\card_generation"
    
    # Crear validador
    validator = PokerDatasetValidator(DATASET_DIR)
    
    # Ejecutar validación
    result = validator.validate_dataset()
    
    # Salir con código de estado apropiado
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()