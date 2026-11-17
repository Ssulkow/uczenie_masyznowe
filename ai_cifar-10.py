import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from datasets import load_dataset
import os

Transformacja = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),                            
    transforms.RandomRotation(15),                                
    transforms.Normalize(mean=(0.49139968,0.48215841,0.44653091),std=(0.24703223,0.24348513,0.26158784))
])
zbior_terningowy = datasets.CIFAR10(root='./data',train=True,download=True,transform=Transformacja)
data_loader = DataLoader(dataset=zbior_terningowy,batch_size=64,shuffle=True)

zbior_testowy = datasets.CIFAR10(root = './data',train=False,download=True,transform=Transformacja)
data_loader_test = DataLoader(dataset=zbior_testowy,batch_size=64,shuffle=False)
data = zbior_terningowy.data/255
mean = data.mean(axis=(0,1,2)) 
std = data.std(axis=(0,1,2))
print(mean)
print(std)
class MojaSiec(nn.Module):
    def __init__(self):
        super(MojaSiec, self).__init__() 
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.MaxPool2d(2,2), #16x16

            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2,2) #8x8
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*8*8,out_features=1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(1024),
            nn.Linear(in_features=1024,out_features=256),
            nn.BatchNorm1d(256),
            nn.PReLU(256),
            nn.Linear(in_features=256,out_features=10),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x
epoka = 0
epoki = 20 
model = MojaSiec()
optymalizator = optim.Adam(model.parameters(),lr=0.01)
kryterium = nn.CrossEntropyLoss()
print("zaczynamy trening...")
for epoka in range( epoki):
    model.eval()
    suma_punktow = 0 
    skutecznosc = 0
    wszystkie_zdjecia = 0
    najlepsza_skutecznosc = 0
    with torch.no_grad():
        for obraz,etykieta in data_loader_test:
            odp = model(obraz)
            zgadywanie_obrazka = torch.argmax(odp,dim=1)
            zdobyte_punkty = (zgadywanie_obrazka==etykieta).sum().item()
            suma_punktow += zdobyte_punkty
            wszystkie_zdjecia += etykieta.size(0)
            skutecznosc = (suma_punktow / wszystkie_zdjecia) * 100
        print(f"Egzamin zakończony!")
        print(f"Sieć zgadła {suma_punktow} na {wszystkie_zdjecia} zdjęć.")
        print(f"Ostateczna skuteczność: {skutecznosc:.2f}%")
        if skutecznosc>najlepsza_skutecznosc:
            najlepsza_skutecznosc = skutecznosc
            Checkpoint = {
                'model_stan': model.state_dict(),
                'model_opti':model.state_dict(),
                'epoka':epoka,
                'najlepsza_skutecznosc':skutecznosc
            }
            torch.save(Checkpoint,'najlepszy_cifar')
            print("pobito rekord: ",epoka," skutecznosc: ", najlepsza_skutecznosc)
        else:
            print("nie pobito rekordu ")
    model.train()
    strata_w_epoce = 0
    for numer_paczki,(obraz,etykieta) in enumerate(data_loader):
        optymalizator.zero_grad()
        predykcje = model(obraz)
        strata = kryterium(predykcje,etykieta)
        strata.backward()
        optymalizator.step()
        strata_w_epoce += strata.item()
        if (numer_paczki+1) % 200 == 0:
            srednia = strata_w_epoce / 200
            print("numer paczki: ",numer_paczki," starta w epoce: ",srednia)




