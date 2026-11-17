import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
# ==========================================
# 1. PRZYGOTOWANIE DANYCH (DataLoader)
# ==========================================
transformacja = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307,std = 0.3081)
])
zbior_treningowy = datasets.MNIST(root='./data', train=True, download=True, transform=transformacja)
dataloader = DataLoader(dataset=zbior_treningowy, batch_size=64, shuffle=True)

zbior_testowy = datasets.MNIST(root = './data',train = False,download=True,transform=transformacja)
dataloader_testowy = DataLoader(zbior_testowy, batch_size=64, shuffle=False)
najlepsza_skutecznosc = 0


# ==========================================
# 2. DEFINICJA SIECI
# ==========================================
class MojaSiec(nn.Module):
    def __init__(self):
       
        super(MojaSiec, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.akt1 = nn.PReLU(32)
        self.cut1 = nn.MaxPool2d(2,2) #14 x 14

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.akt2 = nn.PReLU(64)
        self.cut2 = nn.MaxPool2d(2,2) 

        #7x7
        self.splaszcz = nn.Flatten()
        self.linear = nn.Linear(in_features=64*7*7,out_features=128)
        self.bn1d = nn.BatchNorm1d(128)
        self.aktlin = nn.PReLU(128)
        self.wyjscie = nn.Linear(in_features=128,out_features=10)

    

    def forward(self, x):
     
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.akt1(x)
        x = self.cut1(x)
        
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.akt2(x)
        x = self.cut2(x)

        x = self.splaszcz(x)
        x = self.linear(x)
        x = self.bn1d(x)
        x = self.aktlin(x)
        x = self.wyjscie(x)
        return x
    
model = MojaSiec()
kryterium = nn.CrossEntropyLoss() #funkcja wypsiujac blad
optymalizator = optim.Adam(model.parameters(), lr=0.001)


if os.path.exists('najlepszy_model.pth'):
        Checkpoint = torch.load('najlepszy_model.pth')
        model.load_state_dict(Checkpoint['model_stan'])
        optymalizator.load_state_dict(Checkpoint['optymalizator_stan'])
        najlepsza_skutecznosc = Checkpoint['najlepszy_wynik']
        zapisana_epoka = Checkpoint['epoka']
else:
    print("zaczynamy od zera")
    zapisana_epoka=0

epoki = 100
print("Rozpoczynamy trening...")

for epoka in range(zapisana_epoka,epoki):
    strata_w_epoce = 0.0
    model.eval()
    suma_punktow = 0
    wszystkie_zdjecia = 0
    with torch.no_grad():

        for obrazki,etykiety in dataloader_testowy:
            predykcje = model(obrazki)
            zgadywane_cyfry = torch.argmax(predykcje, dim=1)
            zdobyte_punkty = (zgadywane_cyfry==etykiety).sum().item()
            suma_punktow+= zdobyte_punkty
            wszystkie_zdjecia += etykiety.size(0)
        skutecznosc = (suma_punktow / wszystkie_zdjecia) * 100
        print(f"Egzamin zakończony!")
        print(f"Sieć zgadła {suma_punktow} na {wszystkie_zdjecia} zdjęć.")
        print(f"Ostateczna skuteczność: {skutecznosc:.2f}%")

        if skutecznosc>najlepsza_skutecznosc:
            najlepsza_skutecznosc = skutecznosc

            Checkpoint = {
                'model_stan': model.state_dict(),
                'optymalizator_stan': optymalizator.state_dict(),
                'najlepszy_wynik': najlepsza_skutecznosc,
                'epoka': epoka  # Zapisujemy numer epoki, w której padł rekord
            }

            torch.save(Checkpoint, 'najlepszy_model.pth')
            print(f"Epoka {epoka+1}: Pobito rekord! ({skutecznosc:.2f}%). Zapisano na dysk.")
        else:
            print(f"Epoka {epoka+1}: Brak poprawy ({skutecznosc:.2f}%). Plik bezpieczny.")

    model.train()
    for numer_paczki, (obrazki, etykiety) in enumerate(dataloader):

        # 1. CZYŚCIMY STARE GRADIENTY
        optymalizator.zero_grad()
        
        # 2. PRZEJŚCIE W PRZÓD
        predykcje = model(obrazki)
        
        # 3. OBLICZENIE BŁĘDU
        strata = kryterium(predykcje, etykiety)
        
        # 4. PRZEJŚCIE W TYŁ / AUTOGRAD
        strata.backward()
        
        # 5. AKTUALIZACJA WAG
        optymalizator.step()
        
        # Zapisujemy stratę do wyświetlenia
        strata_w_epoce += strata.item()
        
        # Wyświetlamy status co 200 paczek
        if numer_paczki % 200 == 199:
            srednia_strata = strata_w_epoce / 200
            print(f"Epoka: {epoka+1}, Paczka: {numer_paczki+1}, Błąd (Strata): {srednia_strata:.4f}")
            strata_w_epoce = 0.0
   