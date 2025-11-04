saldo = 0
status = True
def setor():
    global saldo
    jumlah = int(input('Masukkan jumlah setor tunai: '))
    if jumlah > 0:
        saldo += jumlah
        print(f'Saldo Anda sekarang: {saldo}')
    else:
        print('Jumlah setor tunai harus lebih dari 0.')

def tarik():
    global saldo
    jumlah = int(input('Masukkan jumlah tarik tunai: '))
    if jumlah > saldo:
        print('Saldo tidak cukup untuk melakukan penarikan.')
    elif jumlah <= 0:
        print('Jumlah tarik tunai harus lebih dari 0.')
    else:
        saldo -= jumlah
        print(f'Saldo Anda sekarang: {saldo}')

def cek_saldo():
    global saldo
    print(f'Saldo Anda saat ini: {saldo}')

while status == True:
    print('Selamat Datang di ATM Sederhana')
    print('1. Setor Tunai')
    print('2. Tarik Tunai')
    print('3. Cek Saldo')
    print('4. Keluar')

    pilihan = input('Masukkan Pilihan Anda: ')

    if pilihan == '1':
        setor()
        a_input = input('ingin melanjutkan ? (y/n): ')
        if a_input != 'y':
            status = False   
    elif pilihan == '2':
        tarik()
        a_input = input('ingin melanjutkan ? (y/n): ')
        if a_input != 'y':
            status = False  
    elif pilihan == '3':
        cek_saldo()
        a_input = input('ingin melanjutkan ? (y/n): ')
        if a_input != 'y':
            status = False  
    elif pilihan == '4':
        status = False
    else:
        print('Pilihan tidak valid, silakan coba lagi.')

print('Terima kasih telah menggunakan ATM Sederhana!')