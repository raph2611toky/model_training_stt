import argparse
from collections import Counter

def make_gotique(texte):
    texte = texte.lower()
    alphabet = {
        "a":" █████╗ \n██╔══██╗\n███████║\n██╔══██║\n██║  ██║\n╚═╝  ╚═╝",
        "b":"██████╗ \n██╔══██╗\n██████╔╝\n██╔══██╗\n██████╔╝\n╚═════╝ ",
        "c":" ██████╗\n██╔════╝\n██║     \n██║     \n╚██████╗\n ╚═════╝",
        "d":"██████╗ \n██╔══██╗\n██║  ██║\n██║  ██║\n██████╔╝\n╚═════╝ ",
        "e":"███████╗\n██╔════╝\n█████╗  \n██╔══╝  \n███████╗\n╚══════╝",
        "f":"███████╗\n██╔════╝\n███████╗\n██╔════╝\n██║     \n╚═╝     ",
        "g":" ██████╗ \n██╔════╝ \n██║ ████╗\n██║  ██╔╝\n╚█████╔╝ \n ╚════╝  ",
        "h":"██╗  ██╗\n██║  ██║\n███████║\n██╔══██║\n██║  ██║\n╚═╝  ╚═╝",
        "i":"██╗\n██║\n██║\n██║\n██║\n╚═╝",
        "j":"█████████╗\n╚═══██╔══╝\n    ██║   \n██║ ██║   \n╚████╔╝   \n ╚═══╝    ",
        "k":"██╗   ██╗\n██║  ██╔╝\n██████╔╝ \n██╔═██║  \n██║ ╚═██║\n╚═╝   ╚═╝",
        "l":"██╗     \n██║     \n██║     \n██║     \n███████╗\n╚══════╝",
        "m":"██╗      ██╗\n████╗  ████║\n██╔██╗██╗██║\n██║╚███╗ ██║\n██║ ╚══╝ ██║\n╚═╝      ╚═╝",
        "n":"███╗   ██╗\n████╗  ██║\n██╔██╗ ██║\n██║╚██╗██║\n██║ ╚████║\n╚═╝  ╚═══╝",
        "o":" ██████╗ \n██╔═══██╗\n██║   ██║\n██║   ██║\n╚██████╔╝\n ╚═════╝ ",
        "p":"██████╗ \n██╔══██╗\n██████╔╝\n██╔═══╝ \n██║     \n╚═╝     ",
        "q":" ██████╗  \n██╔═══██╗ \n██║   ██║ \n██║ █╗██║ \n╚█████╗██║\n ╚════╝╚═╝",
        "r":"██████╗ \n██╔══██╗\n██████╔╝\n██╔═██║ \n██║  ██║\n╚═╝  ╚═╝",
        "s":"███████╗\n██╔════╝\n╚██████╗\n ╚═══██║\n██████╔╝\n╚═════╝ ",
        "t":"████████╗\n╚══██╔══╝\n   ██║   \n   ██║   \n   ██║   \n   ╚═╝   ",
        "u":"██╗   ██╗\n██║   ██║\n██║   ██║\n██║   ██║\n╚██████╔╝\n ╚═════╝ ",
        "v":"██╗   ██╗\n██╗   ██╗\n██╗   ██╗\n╚██╗ ██╔╝\n ╚████╔╝ \n  ╚═══╝  ",
        "w":"██╗      ██╗\n██║ ███╗ ██║\n██╚██╗██╝██║\n████╗  ████║\n██╔╝    ╚██║\n╚═╝      ╚═╝",
        "x":"██╗   ██╗\n╚██╗ ██╔╝\n ╚████╔╝ \n ██╔═██║ \n██║   ██║\n╚═╝   ╚═╝",
        "y":"██╗   ██╗\n╚██╗ ██╔╝\n  ╚███╔╝ \n   ██╔╝  \n  ██╔╝   \n  ╚═╝    ",
        "z":"████████╗\n     ██╔╝\n   ██╔╝  \n ██╔╝    \n████████╗\n╚═══════╝",
        "1": " ██╗\n███║\n╚██║\n ██║\n ██║\n ╚═╝",
        "2": "██████╗ \n╚════██╗\n █████╔╝\n██╔═══╝ \n███████╗\n╚══════╝",
        "3": "██████╗ \n╚════██╗\n █████╔╝\n ╚═══██╗\n██████╔╝\n╚═════╝ ",
        "4": "██╗  ██╗\n██║  ██║\n███████║\n╚════██║\n     ██║\n     ╚═╝",
        "5": "███████╗\n██╔════╝\n███████╗\n╚════██║\n███████║\n╚══════╝",
        "6": " ██████╗\n██╔════╝\n███████╗\n██╔═══██╗\n╚██████╔╝\n ╚═════╝ ",
        "7": "███████╗\n╚════██║\n    ██╔╝\n   ██╔╝ \n   ██║  \n   ╚═╝  ",
        "8": " █████╗ \n██╔══██╗\n╚█████╔╝\n██╔══██╗\n╚█████╔╝\n ╚════╝ ",
        "9": " █████╗ \n██╔══██╗\n╚██████║\n ╚═══██║\n █████╔╝\n ╚════╝ ",
        " ":"         \n         \n         \n         \n         \n         ",
        "-":"      \n      \n█████╗\n╚════╝\n      \n      ",
        "+":"    ██╗    \n    ██║    \n██████████╗\n╚═══██╔═══╝\n    ██║    \n    ╚═╝    ",
        ",": "      \n      \n      \n      \n ██╗  \n███║  ",
        ".": "      \n      \n      \n      \n ██╗  \n ╚═╝  ",
        "!": " ██╗\n ██║\n ██║\n ╚═╝\n ██║\n ╚═╝",
        "?": "██████╗\n╚════██╗\n  ████╔╝\n  ╚═══╝ \n  ██╗  \n  ╚═╝  "
    }
    all_alphabets = Counter(texte).keys()
    if any(a not in alphabet.keys() for a in all_alphabets):
        print("❌❌ Certaines caractères sont intraduisables")
        exit(1)
    ligne = ["","","","","",""]
    for lettre in texte:
        lettre_gotique = alphabet[lettre.lower()].split("\n")
        for lg in range(len(lettre_gotique)):
            ligne[lg] += lettre_gotique[lg]
    print("\n\n")
    print(*ligne, sep='\n')
    print("\n\n")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Gotique pour les banners.")
    parser.add_argument("--texte","-t", required=True,dest="texte", help="Une caractère à traduire en une gotique.")
    
    args = parser.parse_args()
    make_gotique(args.texte)