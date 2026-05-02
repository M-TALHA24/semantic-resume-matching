import re

def clean_text(text):
    
    text = str(text).lower()

    # remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # normalize separators (important for skills like hr-manager → hr manager)
    text = re.sub(r'[-/|•·]', ' ', text)

    # keep letters, numbers, and IMPORTANT tech symbols (+ # .)
    # C++, C#, .NET, version numbers, years all preserved properly
    text = re.sub(r'[^a-z0-9\s\+\#\.]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text