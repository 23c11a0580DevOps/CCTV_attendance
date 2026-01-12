import fitz
import os
import re

PDF_PATH = "src\\data\\3rd_cse_Student Details.pdf"
OUTPUT_DIR = "src\\data\\pics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HT_PATTERN = re.compile(r"\d{2}C\d{2}A\d{2}[0-9A-Z]{1,2}")

doc = fitz.open(PDF_PATH)

saved = 0

for page in doc:
    blocks = page.get_text("dict")["blocks"]
    current_ht = None

    for block in blocks:
        # -------- TEXT BLOCK --------
        if block["type"] == 0:
            text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"]

            text = text.replace(" ", "")
            match = HT_PATTERN.search(text)
            if match:
                current_ht = match.group()

        # -------- IMAGE BLOCK --------
        elif block["type"] == 1 and current_ht:
            img_bytes = block["image"]
            ext = block.get("ext", "jpg")

            filename = f"{current_ht}.{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(img_bytes)

            print(f"[SAVED] {filename}")
            saved += 1

            current_ht = None  # reset for next row

doc.close()

print("\n✅ EXTRACTION COMPLETE")
print(f"📸 Faces extracted: {saved}")
print(f"📁 Saved in: {OUTPUT_DIR}")
