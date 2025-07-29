import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from functies import *

AGtafel = 'AG2022DefinitiefGevalideerd.xlsx' #Vul hier de filenaam van de AG tafel in
scenarioset = 'cp2022-p-scenarioset-20k-2024q4.xlsx' #Vul hier de filenaam van de scenarioset in

# Data:
Input_pensioenbeleid = pd.read_excel('Input_pensioenbeleid.xlsx', header = None)
# Input_maatmensen = pd.read_excel('Input_maatmensen.xlsx', header = None)
Input_specificaties = pd.read_excel('Input_specificaties.xlsx', header = None)
qx_man = np.asarray(pd.read_excel(AGtafel, sheet_name = 'qx mannen 2022', header = 0, index_col = 0))
qx_vrouw = np.asarray(pd.read_excel(AGtafel, sheet_name = 'qx vrouwen 2022', header = 0, index_col = 0))
MV_verhouding = 0.5
q = (MV_verhouding * qx_man) + ((1 - MV_verhouding) * qx_vrouw)
Ervaringssterfte_dlr = np.asarray(Input_pensioenbeleid.iloc[5:108])[:,1]
Ervaringssterfte_ptr = np.asarray(Input_pensioenbeleid.iloc[109:212])[:,1]
# MV_verhouding = np.asarray(Input_pensioenbeleid.iloc[213:316])[:,1]
duratie = np.asarray(Input_pensioenbeleid.iloc[421:])[:,1]
alpha = np.asarray(Input_pensioenbeleid.iloc[317:420])[:,1]
opslag = np.asarray(Input_specificaties.iloc[209:])[:,1] # Andere opties: 105:208, 1:104

with st.form("my_form"):
    Franchise = st.number_input('Franchise per jaar:', 0, 1000000000000)    
    VasteKosten = st.number_input('Vaste kosten per jaar (euro):', 0, 1000000)
    VermogensKosten = st.number_input('Vermogenskosten per jaar (%):', 0, 100)/100
    Pensioenleeftijd = st.number_input('Pensioenleeftijd:', 1, 100)
    StartSalaris = st.number_input('StartSalaris jaar:', 0, 10000000)
    n_scenarios = st.number_input('Scenario aantal:', 1, 5000)
    lft = st.number_input('Leeftijd deelnemer:', 1, 100)
    V_0 = st.number_input('Startvermogen:', 0, 1000000000000)
    PremiePercentage = st.number_input('Premiepercentage:', 0, 100)

    submitted = st.form_submit_button("Submit")
    

# Scenarioset input laden
Aandelenrendement = np.asarray(pd.read_excel(scenarioset, sheet_name = '4_Aandelenrendement', header = None))
X1 = np.asarray(pd.read_excel(scenarioset, sheet_name = '1_Toestandsvariabele_1', header = None))
X2 = np.asarray(pd.read_excel(scenarioset, sheet_name = '2_Toestandsvariabele_2', header = None))
X3 = np.asarray(pd.read_excel(scenarioset, sheet_name = '3_Toestandsvariabele_3', header = None))
phi_N = np.asarray(pd.read_excel(scenarioset, sheet_name = '7_Renteparameter_phi_N', header = None))
Psi_N = np.asarray(pd.read_excel(scenarioset, sheet_name = '8_Renteparameter_Psi_N', header = None))

min_lft = 18
n_jaren = X1.shape[1]
AOW = 1580 * 12

# Salaris ontwikkeling:
loonverloop = np.zeros(len(opslag))
loonverloop[0] = StartSalaris

for i in range(len(opslag) - 1):
    # loonverloop[i + 1] = loonverloop[i] * (1 + opslag[i])
    loonverloop[i + 1] = loonverloop[i] * 1.03

franchiseverloop = np.zeros(len(opslag))
franchiseverloop[0] = Franchise

for i in range(len(opslag) - 1):
    franchiseverloop[i + 1] = franchiseverloop[i] * 1.02

# # Functie die rentetermijnstructuur berekent
# def bereken_rentetermijnstructuur(s, tau, t, phi, Psi, X1, X2, X3):
#     return np.exp(-(1/(tau + 1)) * (phi[tau, t] + Psi[tau, 0] * X1[s, t] + Psi[tau, 1] * X2[s, t] + Psi[tau, 2] * X3[s, t])) - 1
    
# # Functie die rendemenet op obligaties berekent
# def bereken_rendement_obligaties(s, d, t, r):
#     return (((1 + r[s, d, t])**(d)) / ((1 + r[s, d - 1, t + 1])**(d-1))) - 1

# # Functie die vermogen per projectiejaar berekent
# def V(V_t, premie, uitkering, alpha, Aandelenrendement, rendement_obligaties, s, t, p):
#     return ((V_t + premie - uitkering) * (alpha * (1 + Aandelenrendement[s, t]) + 
#                                            (1 - alpha) * (1 + rendement_obligaties[s, t]))) / p
# # Functie die <a> berekent
# def bereken_a(lft, t, Pensioenleeftijd, q, r, s, t0 = 0): #Duidelijkere naam geven dan 'a'
#     a = 1
#     p = 1
#     for n in range(0, 100 - lft - t): #Checken dat dit klopt qua indexing!
#         p = p * (1 - q[lft + t + n, t + n + t0])
#         if (lft + t + n > Pensioenleeftijd):
#             noemer = ((1 + r[s, n, t + t0])**n)
#             a += p / noemer
#     return a

# #Plot waaier obv vermogensverloop en parameters
# def plot_waaier(vermogen, Pensioenleeftijd, Percentielen):
#     scenariobedragen_opbouw = vermogen[:, 0:(Pensioenleeftijd)]
#     kolom_sort = np.take_along_axis(scenariobedragen_opbouw, np.argsort(scenariobedragen_opbouw, axis = 0), axis = 0)
#     x = np.arange(0, kolom_sort.shape[1], 1)
#     UPOwaardes = np.zeros([kolom_sort.shape[1], len(Percentielen)])
#     for i in range(len(Percentielen)):
#         y = kolom_sort[int(kolom_sort.shape[0] * Percentielen[i])]
#         UPOwaardes[:, i] = y
#         plt.plot(x, y)
#     plt.grid()
#     plt.ylabel('Vermogen')
#     plt.title('Vermogenspercentielen per jaar')
#     plt.legend(['Slecht', 'Midden', 'Goed'])
#     return UPOwaardes

# #Bereken verloop van vermogen obv o.a. rentetermijnstructuren, rendementen op obligaties. 
# # Optioneel argument t0 verzet Aandelenrendement en rendement_obligaties
# def verloop_vermogen(rentetermijnstructuren, Aandelenrendement, rendement_obligaties, loonverloop, franchiseverloop,
#                      alpha, q, Pensioenleeftijd, lft, n_jaren, n_scenarios, V_0, Premie_percentage, VasteKosten, VermogensKosten, t0 = 0):
#     # Initiele waardes
#     vermogen = np.zeros([n_scenarios, n_jaren])
#     vermogen[:, lft - 1] = V_0
#     uitkeringen = np.zeros([n_scenarios, n_jaren])
#     for s in range(n_scenarios):
#         V_t = V_0
#         for t in range(n_jaren - lft):
#             if Gebruik_Ervaringssterfte:
#                 p = 1 - q[lft + t, t] * Ervaringssterfte_dlr[lft + t]
#             else:
#                 p = 1 - q[lft + t, t]
#             alph = alpha[lft + t - min_lft]
#             if lft + t >= Pensioenleeftijd:
#                 a = bereken_a(lft, t, Pensioenleeftijd, q, rentetermijnstructuren, s)
#                 uitkering = V_t/a
#                 premie = -VasteKosten
#                 V_t = V(V_t, premie, uitkering, alph, Aandelenrendement[:, t0:], rendement_obligaties[:, t0:], s, t, p) * (1 - VermogensKosten)
#             else: 
#                 a = bereken_a(lft, t, Pensioenleeftijd, q, rentetermijnstructuren, s)
#                 uitkering = V_t/a
#                 premie = Premie_percentage * (loonverloop[t + t0] - franchiseverloop[t + t0]) - VasteKosten
#                 V_t = V(V_t, premie, 0, alph, Aandelenrendement[:, t0:], rendement_obligaties[:, t0:], s, t, p) * (1 - VermogensKosten)
#             # Sla vermogens en uitkeringen op
#             vermogen[s, t + lft] = V_t
#             uitkeringen[s, t + lft] = uitkering
#     return vermogen, uitkeringen

# Voer bovenstaande functie uit per scenario, looptijd en projectiejaar
file_path = f'rentetermijn_nominaal_{n_scenarios}_{scenarioset}.npy'

if os.path.exists(file_path):
    rentetermijnstructuren_nominaal = np.load(file_path)
else:
    rentetermijnstructuren_nominaal = np.zeros([n_scenarios, phi_N.shape[0], X1.shape[1]]) #[s, tau, t]
    for s in range(n_scenarios):
        for tau in range(phi_N.shape[0]):
            for t in range(X1.shape[1]):
                rentetermijnstructuren_nominaal[s, tau, t] = bereken_rentetermijnstructuur(s, tau, t, phi_N, Psi_N, X1, X2, X3)
    np.save(file_path, rentetermijnstructuren_nominaal)

# Bereken rendement op obligaties per scenario en projectiejaar, tenzij het resultaat al vastligt in een file?
rendement_obligaties = np.zeros([n_scenarios, X1.shape[1]])
for s in range(n_scenarios):
    for t in range(X1.shape[1] - (lft - 18)):
        d = int(duratie[(lft - 18) + t])
        rendement_obligaties[s, t] = bereken_rendement_obligaties(s, d, t, rentetermijnstructuren_nominaal)

# Initiele waardes
t0 = 0
vermogen = np.zeros([n_scenarios, n_jaren])
ps = np.zeros([n_scenarios, n_jaren])
premies = np.zeros([n_scenarios, n_jaren])
uitkeringen = np.zeros([n_scenarios, n_jaren])
atjes = np.zeros([n_scenarios, n_jaren])

vermogen[:, lft - 1] = V_0
uitkeringen = np.zeros([n_scenarios, n_jaren])
for s in range(n_scenarios):
    V_t = V_0
    for t in range(n_jaren - lft):
        p = 1 - q[lft + t, t] * Ervaringssterfte_dlr[lft + t]
        alph = alpha[lft + t - min_lft]

        if lft + t >= Pensioenleeftijd:
            a = bereken_a(lft, t, Pensioenleeftijd, q, rentetermijnstructuren_nominaal, s, t0)
            atjes[s, t] = a
            uitkering = V_t/a
            premie = -VasteKosten
            V_t = V(V_t, premie, uitkering, alph, Aandelenrendement[:, t0:], rendement_obligaties[:, t0:], s, t, p) * (1 - VermogensKosten)
        else: 
            a = bereken_a(lft, t, Pensioenleeftijd, q, rentetermijnstructuren_nominaal, s, t0)
            atjes[s, t] = a
            uitkering = V_t/a
            premie = PremiePercentage * (loonverloop[t + t0] - franchiseverloop[t + t0]) - VasteKosten
            V_t = V(V_t, premie, 0, alph, Aandelenrendement[:, t0:], rendement_obligaties[:, t0:], s, t, p) * (1 - VermogensKosten)
        # Sla vermogens en uitkeringen op
        vermogen[s, t + lft] = V_t
        uitkeringen[s, t + lft] = uitkering

nav_vermogen = plot_waaier(vermogen, Pensioenleeftijd, [0.05, 0.5, 0.95])

data_vermogen = pd.DataFrame(nav_vermogen, columns = ["Slecht weer", "Verwacht", "Goed weer"])

st.line_chart(data_vermogen, x_label = "Leeftijd deelnemer", y_label = "Geprojecteerd vermogen")

nav_uitkeringen = plot_waaier(uitkeringen, Pensioenleeftijd, [0.05, 0.5, 0.95])

data_uitkeringen = pd.DataFrame(nav_uitkeringen, columns = ["Slecht weer", "Verwacht", "Goed weer"])

st.line_chart(data_uitkeringen, x_label = "Leeftijd deelnemer", y_label = "Geprojecteerde uitkering")