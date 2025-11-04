peserta = [
    ['Andi', 90],['Budi', 85],['Cici', 95],['Dodi', 80],['Eka', 88],['Fani', 92],['Gina', 87],['Hani', 93],['Ika', 89],['Joni', 91],
    ['Kiki', 84],['Lina', 96],['Mila', 82],['Nina', 94],['Oki', 86],['Puti', 81],['Qori', 83],['Rina', 97],['Siti', 90],['Tina', 88] 
]

diurutkan = sorted(peserta,key = lambda x: x[1], reverse=True)
pemenang = diurutkan[:3]

for x in range(3):
    print(f'Juara-{x+1} : {pemenang[x][0]} , Dengan Waktu : {pemenang[x][1] // 60} Jam {pemenang[x][1] % 60} Menit')