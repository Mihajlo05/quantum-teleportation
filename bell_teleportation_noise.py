from multiprocessing import Value
import qutip as qt
import matplotlib.pyplot as plt
import math
import numpy as np
from enum import Enum

##########################################################################
##############Korisne varijable i funkcije################################
##########################################################################




# korisne varijable
ket0 = qt.ket([0])
ket1 = qt.ket([1])
ket00 = qt.tensor(ket0, ket0)
ket11 = qt.tensor(ket1, ket1)
ket01 = qt.tensor(ket0, ket1)
ket10 = qt.tensor(ket1, ket0)
identity = qt.identity(2)
########################################################################

#######operatori suma##############################
def get_bit_flip_operators(probability_of_error):
    operators = []
    operators.append(math.sqrt(1 - probability_of_error) * identity)
    operators.append(math.sqrt(probability_of_error) * qt.sigmax())
    return operators

def get_phase_flip_operators(probability_of_error):
    operators = []
    operators.append(math.sqrt(1 - probability_of_error) * identity)
    operators.append(math.sqrt(probability_of_error) * qt.sigmaz())
    return operators

def get_depolarizing_noise_operators(probability_of_error):
    operators = []
    p = probability_of_error
    operators.append(math.sqrt(1 - ((3*probability_of_error)/4)) * identity)
    operators.append(math.sqrt(p/4) * qt.sigmax())
    operators.append(math.sqrt(p/4) * qt.sigmay())
    operators.append(math.sqrt(p/4) * qt.sigmaz())
    return operators
###################################################


#sum

class NoiseMode(Enum):
    NoNoise = 0
    BitFlip = 1
    PhaseFlip = 2
    DepolarizingNoise = 3

def get_noise_operators(prob_of_error, noise_mode):
    if noise_mode == NoiseMode.NoNoise:
        return []
    elif noise_mode == NoiseMode.BitFlip:
        return get_bit_flip_operators(prob_of_error)
    elif noise_mode == NoiseMode.PhaseFlip:
        return get_phase_flip_operators(prob_of_error)
    elif noise_mode == NoiseMode.DepolarizingNoise:
        return get_depolarizing_noise_operators(prob_of_error)


##########################################################################
##########################################################################
##########################################################################



#pocetno kvantno stanje
a_in = 1 / math.sqrt(3)
b_in = math.sqrt(1 - a_in**2)
psi_in = a_in * ket0 + b_in * ket1
ro_in = psi_in * psi_in.dag()


#varijabla pomocu koje se odredjuje stanje u kome se nalazi kvantni kanal (ch = channel)
#u kvantni kanal se nalaze dva uvezana qubita, jedan kod alis, a drugi kod boba,
#i preko njega se teleportuje alisino stanje (_in)
theta_ch = math.pi / 4

psi_ch = math.sin(theta_ch) * ket00 + math.cos(theta_ch) * ket11 #kvantno stanje kanala
ro_ch = psi_ch * psi_ch.dag() #matrica gustine kanala

#matrica gustine koja opisuje celokupno kvantno stanje
ro = qt.tensor(ro_in, ro_ch)

##baze po kojima se mere pocetno stanje i jedan qubit iz kvantnog kanala##########
phi_proj = theta_ch
sinphi = math.sin(phi_proj)
cosphi = math.cos(phi_proj)

b1 = cosphi * ket00 + sinphi * ket11
b2 = sinphi * ket00 - cosphi * ket11
b3 = cosphi * ket01 + sinphi * ket10
b4 = sinphi * ket01 - cosphi * ket10
#################################################################################

###matrice projektivnog merenja (na osnovu baza merenja izracunata u prethodnoj sekciji)
p1 = (b1 * b1.dag())
p2 = (b2 * b2.dag())
p3 = (b3 * b3.dag())
p4 = (b4 * b4.dag())

# kako se ove matrice primenjuju na matricu gustine celokupnog kvantnog stanja, potrebno je da se
# matrice projekcije pomnoze tenzorski sa identitetom kako bi se poklapale dimenzije (a zbog identiteta, treci qubit
# ostaje nepromenjen primenjivanjem ovih matrica)
p1 = qt.tensor(p1, identity)
p2 = qt.tensor(p2, identity)
p3 = qt.tensor(p3, identity)
p4 = qt.tensor(p4, identity)
##############################################################################################################

###matrice korekcije koje primalac qubita primenjuje kako bi dobio isto kvantno stanje kao pocetno##
u1 = qt.identity(2)
u2 = qt.sigmaz()
u3 = qt.sigmax()
u4 = qt.sigmaz() * qt.sigmax()
#####################################################################################################

###verovatnoce merenja odredjenog stanja#########
q1 = (p1 * ro).tr()
q2 = (p2 * ro).tr()
q3 = (p3 * ro).tr()
q4 = (p4 * ro).tr()
##################################################

n_of_teleportations = 100
error_prob_values = np.linspace(0, 1, n_of_teleportations)

# podesi ove varijable da bi izabrao tip suma za odrenjenu qubit
noise_in_mode = NoiseMode.PhaseFlip
noise_ch1_mode = NoiseMode.BitFlip
noise_ch2_mode = NoiseMode.DepolarizingNoise

fidelity_values = []

for error_prob in error_prob_values:
    ro_noise = ro

    temp = None
    for operator in get_noise_operators(error_prob, noise_ch2_mode): #primenjivanje suma na trecem qubitu
        oper = qt.tensor(identity, identity, operator)
        value = oper * ro_noise * oper.dag()

        if temp is None:
            temp = value
        else:
            temp += value
    if temp is not None:
        ro_noise = temp

    temp = None
    for operator in get_noise_operators(error_prob, noise_ch1_mode): #primenjivanje suma na drugom qubitu
        oper = qt.tensor(identity, operator, identity)
        value = oper * ro_noise * oper.dag()

        if temp is None:
            temp = value
        else:
            temp += value
    if temp is not None:
        ro_noise = temp

    temp = None
    for operator in get_noise_operators(error_prob, noise_in_mode): #primenjivanje suma na prvom qubitu
        oper = qt.tensor(operator, identity, identity)
        value = oper * ro_noise * oper.dag()

        if temp is None:
            temp = value
        else:
            temp += value
    if temp is not None:
        ro_noise = temp
    

    varrho1 = u1 * (p1 * ro_noise * p1).ptrace(2) * u1.dag()
    varrho1 /= q1
    varrho2 = u2 * (p2 * ro_noise * p2).ptrace(2) * u2.dag()
    varrho2 /= q2
    varrho3 = u3 * (p3 * ro_noise * p3).ptrace(2) * u3.dag()
    varrho3 /= q3
    varrho4 = u4 * (p4 * ro_noise * p4).ptrace(2) * u4.dag()
    varrho4 /= q4

    f1 = (ro_in * varrho1).tr()
    f2 = (ro_in * varrho2).tr()
    f3 = (ro_in * varrho3).tr()
    f4 = (ro_in * varrho4).tr()

    f = q1 * f1 + q2 * f2 + q3 * f3 + q4 * f4
    fidelity_values.append(f)

plt.figure()
plt.xlim(0, 1.01)
plt.ylim(0, 1.01)
plt.xlabel("verovatnoca greske")
plt.ylabel("vernost stanja")
plt.plot(error_prob_values, fidelity_values)
plt.show()