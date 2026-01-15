import cv2
import pytesseract
import numpy as np
import os
from PIL import Image

# --- CONFIGURATION ---
# Assure-toi que le chemin est bon pour ta machine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def load_image_robust(path):
    """Charge une image de manière robuste."""
    if not os.path.exists(path):
        print(f"[ERREUR] Fichier introuvable : {path}")
        return None

    img = cv2.imread(path)
    if img is not None: return img
    
    print("[INFO] Chargement via Pillow (fallback)...")
    try:
        pil_img = Image.open(path).convert('RGB')
        img = np.array(pil_img)
        img = img[:, :, ::-1].copy()
        return img
    except Exception as e:
        print(f"[ERREUR] Échec chargement : {e}")
        return None

def filter_noise_by_contours(binary_img, min_area=50):
    """Supprime les petits points (bruit) via findContours."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_img = binary_img.copy()
    noise_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            cv2.drawContours(cleaned_img, [cnt], -1, 0, thickness=cv2.FILLED)
            noise_count += 1
            
    return cleaned_img

def auto_invert_for_tesseract(img):
    """
    Vérifie si le fond est noir ou blanc et s'assure que Tesseract 
    reçoit TOUJOURS du texte noir sur fond blanc.
    """
    white_pixels = np.sum(img > 127)
    black_pixels = np.sum(img <= 127)

    if black_pixels > white_pixels:
        print("[AUTO] Fond Noir détecté -> Inversion pour l'OCR")
        return cv2.bitwise_not(img)
    else:
        print("[AUTO] Fond Blanc détecté -> On laisse tel quel")
        return img

def process_smart_pipeline(path, output_path=None, min_contour_area=50):
    # print(f"--- Traitement : {path} ---") # Removed to reduce spam
    
    # 1. Chargement
    original = load_image_robust(path)
    if original is None: return

    # 2. Gris + Resize
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    proc_img = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. Seuillage
    thresh = cv2.adaptiveThreshold(proc_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 19, 5)

    # 4. Nettoyage Bruit
    img_clean = filter_noise_by_contours(thresh, min_area=min_contour_area)

    # 5. Fermeture
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morph = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

    # 6. NORMALISATION INTELLIGENTE
    final_img = auto_invert_for_tesseract(img_morph)

    # --- SAUVEGARDE DE L'IMAGE ---
    if output_path:
        cv2.imwrite(output_path, final_img)
        # print(f"[SUCCÈS] Saved: {output_path}")
    # -----------------------------

    # 7. OCR (Optional now, mainly for dataset creation)
    # config = r'--psm 8 -c tessedit_char_whitelist=lzT'
    # text = pytesseract.image_to_string(final_img, config=config)
    # clean_text = text.strip().replace(" ", "").replace("\n", "")
    # print(f">>> RÉSULTAT OCR : {clean_text}")

# --- MAIN ---
if __name__ == "__main__":
    import glob
    from pathlib import Path

    # DOSSIIER INPUT (Dataset j)
    # Utilisation de doubles backslashes ou raw strings pour Windows
    INPUT_DIR = r"new/path"
    OUTPUT_DIR = "dataset_NB" # Créé dans le dossier courant

    # Création du dossier de sortie s'il n'existe pas
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Dossier '{OUTPUT_DIR}' créé.")
    else:
        print(f"Dossier '{OUTPUT_DIR}' existe déjà.")

    # List files
    files = glob.glob(os.path.join(INPUT_DIR, "*.gif"))
    print(f"Validation : {len(files)} images trouvées dans {INPUT_DIR}")

    count = 0
    for file_path in files:
        # Nom de fichier original
        filename = os.path.basename(file_path)
        
        # Changer l'extension en .png
        filename_png = os.path.splitext(filename)[0] + ".png"
        
        # Chemin de sortie
        output_path_full = os.path.join(OUTPUT_DIR, filename_png)
        
        # Traitement
        # J'ai mis 500 pour enlever les gros morceaux de bruit, ajuste si ça efface les lettres
        # Note: Reduced print inside function to keep console clean
        process_smart_pipeline(file_path, output_path=output_path_full, min_contour_area=700)
        
        count += 1
        if count % 100 == 0:
            print(f"Traités : {count}/{len(files)}")
            
    print(f"Terminé ! {count} images traitées et enregistrées dans '{OUTPUT_DIR}'.")