import PyPDF2
import sys
import os

def extract_text_from_pdf(pdf_path, output_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            full_text = []
            full_text.append(f"Source: {pdf_path}\n")
            full_text.append(f"Total pages: {num_pages}\n")
            full_text.append("=" * 80 + "\n")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                full_text.append(f"\n{'=' * 80}\n")
                full_text.append(f"PAGE {page_num + 1}\n")
                full_text.append(f"{'=' * 80}\n\n")
                full_text.append(text)
                full_text.append("\n")
            
            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(''.join(full_text))
            
            print(f"Successfully extracted text from {num_pages} pages")
            print(f"Output saved to: {output_path}")
            
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        pdf_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        # Fallback to hardcoded paths if no args provided
        pdf_path = r"c:\Users\Gianne Bacay\Desktop\project test\gpbacay_arcane\docs\VL-JEPA Joint Embedding Predictive Architecture for Vision-language.pdf"
        output_path = r"c:\Users\Gianne Bacay\Desktop\project test\gpbacay_arcane\docs\VLJEPA_extracted.txt"
    
    extract_text_from_pdf(pdf_path, output_path)
