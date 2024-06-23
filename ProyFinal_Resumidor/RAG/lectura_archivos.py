import os
import docx2pdf
import pptxtopdf
from fpdf import FPDF
import shutil
import pytesseract
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pythoncom  # Importar la librería COM

def txt_to_pdf(nombre, ubicacion):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    with open(nombre, "r", encoding='utf-8') as f:
        for x in f:
            pdf.multi_cell(0, 10, txt=x)
    
    pdf.output(ubicacion + "/" + os.path.splitext(nombre)[0] + '.pdf')

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def save_text_to_pdf(text, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    lines = text.split('\n')
    
    y = height - 40
    for line in lines:
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line)
        y -= 14

    c.save()

def convert_image_to_pdf(image_path, output_folder, name):
    text = extract_text_from_image(image_path)
    pdf_path = output_folder + '/' + name + '.pdf'
    save_text_to_pdf(text, pdf_path)
    print(f"Texto almacenado en {pdf_path}")

def load_files(ejemplo_dir, output_folder="./documents/"):
    try:
        # Inicializar COM
        pythoncom.CoInitialize()
        
        with os.scandir(output_folder) as output_files:
            for file in output_files:
                file_path = os.path.join(output_folder, file.name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        with os.scandir(ejemplo_dir) as ficheros:
            for fichero in ficheros:
                input_file = ejemplo_dir + '/' + fichero.name

                if fichero.name.endswith('.docx'):
                    docx2pdf.convert(input_file, output_folder)

                elif fichero.name.endswith('.pptx'):
                    pptxtopdf.convert(input_file, output_folder)

                elif fichero.name.endswith('.txt'):
                    txt_to_pdf(input_file, output_folder)

                elif fichero.name.endswith('.pdf'):
                    shutil.copy2(input_file, output_folder)

                elif fichero.name.endswith('.jpg'):
                    nombre_sin_extension, _ = os.path.splitext(fichero.name)
                    convert_image_to_pdf(input_file, output_folder, nombre_sin_extension)
    
    except Exception as e:
        print(f"Error al cargar archivos: {str(e)}")
    
    finally:
        # Desinicializar COM
        pythoncom.CoUninitialize()
