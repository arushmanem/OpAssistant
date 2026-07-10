from document_loader import load_pdf

text = load_pdf('ArushManem_Resume_Updated copy.docx.pdf')

print('=== GPA area ===')
idx = text.find('GPA')
if idx >= 0:
    print(repr(text[idx:idx+30]))

print('=== Random Forest area ===')
for keyword in ['Random', 'random', 'Forest', 'forest']:
    idx = text.find(keyword)
    if idx >= 0:
        print(f'Found {keyword!r} at index {idx}: {text[idx:idx+50]!r}')
        break

print('=== Spider area ===')
for keyword in ['Spider', 'spider', 'Verse', 'verse']:
    idx = text.find(keyword)
    if idx >= 0:
        print(f'Found {keyword!r} at index {idx}: {text[idx:idx+50]!r}')
        break