import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def extend_bounding_rects(img, components):

    extended_rects = []

    for component in components:
        (y,x) = component[:2]
        x_padding = 5 #int(x.stop * 0.05)
        y_padding = 5 #int(y.stop * 0.05)

        x_start = max(0, x.start - x_padding)
        y_start = max(0, y.start - y_padding)
        x_stop = min(img.shape[1], x.stop + 2*x_padding)
        y_stop = min(img.shape[0], y.stop + 2*y_padding)

        roi = img[y_start:y_stop, x_start:x_stop]
        extended_rects.append(roi)

    return extended_rects

def ocr_on_bounding_boxes(img, components):

    texts = []
    extended_rects = extend_bounding_rects(img, components)

    lang = 'jpn'
    oem = 0
    psm = 5
    config = f"-l {lang} --oem {oem} --psm {psm}"
    
    for rect in extended_rects:
        text = pytesseract.image_to_string(rect, config=(config))
        text = text.replace('\f','')
        texts.append(text)

    with open("./test/output.txt", "w", encoding='utf-8') as txt_file:
        for text in texts:
            txt_file.write(" ".join(text) + "\n")

    return texts