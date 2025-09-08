import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        detector = chardet.universaldetector.UniversalDetector()
        for line in file:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']


file_path = r'./src/data/quant-modded-docs.json'
#file_path = r'./src/data/Docs.json'
print(f'The encoding of the file is: {detect_encoding(file_path)}')