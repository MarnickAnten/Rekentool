import numpy as np
import matplotlib.pyplot as plt

# Functie die rentetermijnstructuur berekent
def bereken_rentetermijnstructuur(s, tau, t, phi, Psi, X1, X2, X3):
    return np.exp(-(1/(tau + 1)) * (phi[tau, t] + Psi[tau, 0] * X1[s, t] + Psi[tau, 1] * X2[s, t] + Psi[tau, 2] * X3[s, t])) - 1
    
# Functie die rendemenet op obligaties berekent
def bereken_rendement_obligaties(s, d, t, r):
    return (((1 + r[s, d, t])**(d)) / ((1 + r[s, d - 1, t + 1])**(d-1))) - 1

# Functie die vermogen per projectiejaar berekent
def V(V_t, premie, uitkering, alpha, Aandelenrendement, rendement_obligaties, s, t, p):
    return ((V_t + premie - uitkering) * (alpha * (1 + Aandelenrendement[s, t]) + 
                                           (1 - alpha) * (1 + rendement_obligaties[s, t]))) / p
# Functie die <a> berekent
def bereken_a(lft, t, Pensioenleeftijd, q, r, s, t0 = 0): #Duidelijkere naam geven dan 'a'
    a = 1
    p = 1
    for n in range(0, 100 - lft - t): #Checken dat dit klopt qua indexing!
        p = p * (1 - q[lft + t + n, t + n + t0])
        if (lft + t + n > Pensioenleeftijd):
            noemer = ((1 + r[s, n, t + t0])**n)
            a += p / noemer
    return a

#Plot waaier obv vermogensverloop en parameters
def plot_waaier(vermogen, Pensioenleeftijd, lft, Percentielen):
    scenariobedragen_opbouw = vermogen[:, 0:(Pensioenleeftijd)]
    kolom_sort = np.take_along_axis(scenariobedragen_opbouw, np.argsort(scenariobedragen_opbouw, axis = 0), axis = 0)
    x = np.arange(0, kolom_sort.shape[1], 1)
    UPOwaardes = np.zeros([kolom_sort.shape[1], len(Percentielen)])
    for i in range(len(Percentielen)):
        y = kolom_sort[int(kolom_sort.shape[0] * Percentielen[i])]
        UPOwaardes[:, i] = y
        plt.plot(x, y)
    plt.grid()
    plt.ylabel('Vermogen')
    plt.title('Vermogenspercentielen per jaar')
    plt.legend(['Slecht', 'Midden', 'Goed'])
    return UPOwaardes

#Bereken verloop van vermogen obv o.a. rentetermijnstructuren, rendementen op obligaties. 
# Optioneel argument t0 verzet Aandelenrendement en rendement_obligaties
def verloop_vermogen(rentetermijnstructuren, Aandelenrendement, rendement_obligaties, loonverloop, franchiseverloop,
                     alpha, q, Pensioenleeftijd, lft, n_jaren, n_scenarios, V_0, Premie_percentage, VasteKosten, VermogensKosten, t0 = 0):
    # Initiele waardes
    vermogen = np.zeros([n_scenarios, n_jaren])
    vermogen[:, lft - 1] = V_0
    uitkeringen = np.zeros([n_scenarios, n_jaren])
    for s in range(n_scenarios):
        V_t = V_0
        for t in range(n_jaren - lft):
            p = 1 - q[lft + t, t + t0] * Ervaringssterfte_dlr[lft + t]
            alph = alpha[lft + t - min_lft]
            if lft + t >= Pensioenleeftijd:
                a = bereken_a(lft, t, Pensioenleeftijd, q, rentetermijnstructuren, s)
                uitkering = V_t/a
                premie = -VasteKosten
                V_t = V(V_t, premie, uitkering, alph, Aandelenrendement[:, t0:], rendement_obligaties[:, t0:], s, t, p) * (1 - VermogensKosten)
            else: 
                a = bereken_a(lft, t, Pensioenleeftijd, q, rentetermijnstructuren, s)
                uitkering = V_t/a
                premie = Premie_percentage * (loonverloop[t] - franchiseverloop[t]) - VasteKosten # Idee hier is dat 'startloon' op 'lft' is.
                V_t = V(V_t, premie, 0, alph, Aandelenrendement[:, t0:], rendement_obligaties[:, t0:], s, t, p) * (1 - VermogensKosten)
            # Sla vermogens en uitkeringen op
            vermogen[s, t + lft] = V_t
            uitkeringen[s, t + lft] = uitkering
    return vermogen, uitkeringen