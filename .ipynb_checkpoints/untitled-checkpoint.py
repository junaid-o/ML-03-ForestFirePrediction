import glob

files = glob.glob("/*.html")
html_list = []

for file in files:
    with open(file, "r") as f:
        html_list.append(f.read())
html_str = "\n".join(html_list)
