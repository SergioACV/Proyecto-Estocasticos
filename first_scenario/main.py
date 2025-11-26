from lib.simulation import Simulator
from lib.epidemic_model import EpidemicModel

#PARAMETERS
TOTAL_TIME = 168.0 #hours
dt = 1.0 #1 hour

#INITIAL CONDITIONS
N = 10000 # nodes on all network
I0 = 15 # nodes initially infected
S0 = N-I0 # nodes initially susceptible

#rates
beta = 1.62
r = 2

assert(I0<=N)

#INITIAL STATE [S,I]
initial_state = [S0,I0]

#Model
model = EpidemicModel(beta,r)

#Simulator
simulator = Simulator(model,initial_state,dt,TOTAL_TIME)

simulator.run()



