import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def getAcc(pos, mass, G, softening):
	"""
	Calcula la aceleración para cada partícula 
	pos es una matriz de N x 3 matrix con las posiciones
	mass is un vector N x 1 de masas
	G es la constante gravitacional
	a es una madtriz de N x 3 matrix de aceleraciones
	"""
	# posiciones r = [x,y,z] para cada partícula
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]
	
	# matriz que guarda todas las separaciones entre partículas: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z
	
	# matrix que guarda 1/r^3 para cada par de partículas
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)**(-1.5)
	
	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass
	
	# juntar los componentes de aceleración
	a = np.hstack((ax,ay,az))
	
	return a

def getEnergy( pos, vel, mass, G ):
	"""
	Calcula la energía potencial PE y la energía cinética KE para las partículas
	"""
	# Energía cinética
	KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    
	# posiciones de cada partícula
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	dx = x.T - x
	dy = y.T - y
	dz = z.T - z
 
	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]
    
    # Energía potencial

	PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))

	return KE, PE;
	
# Parámetros de simulación
N         = 100    # número de partículas
t         = 0      # tiempo actual de la simulación
tEnd      = 10.0   # tiempo para el cual la simulación termina
dt        = 0.01   # salto temporal
softening = 0.1    # softening
G         = 1    # constante gravitacional
	
# Generar condiciones iniciales
np.random.seed(17) # seed para la generación de números aleatorios
	
mass = 20.0*np.ones((N,1))/N  # la masa total de las partículas es 30
pos  = np.random.randn(N,3)   # posiciones y velocidades aleatorias
vel  = np.random.randn(N,3)
	
	# se toma la velocidad en relación al centro de masa
vel -= np.mean(mass * vel,0) / np.mean(mass)
	
	# aceleraciones gravitacionales iniciales
acc = getAcc( pos, mass, G, softening )
	
	# energía inicial del sistema
KE, PE  = getEnergy( pos, vel, mass, G )
	
	# número de saltos temporales
Nt = int(np.ceil(tEnd/dt))
	
	# se guardan las energías en un arreglo
pos_save = np.zeros((N,3,Nt+1))
pos_save[:,:,0] = pos
KE_save = np.zeros(Nt+1)
KE_save[0] = KE
PE_save = np.zeros(Nt+1)
PE_save[0] = PE
t_all = np.arange(Nt+1)*dt
	
    #se crean los objetos de figura
fig = plt.figure(figsize=(4,5), dpi=80)
grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
ax1 = plt.subplot(grid[0:2,0])
ax2 = plt.subplot(grid[2,0])

	# loop para la simulación
for i in range(Nt):
	# (1/2) kick
	vel += acc * dt/2.0
		
		# drift
	pos += vel * dt
		
		# se actualizan las aceleraciones
	acc = getAcc( pos, mass, G, softening )
		
		# (1/2) kick
	vel += acc * dt/2.0
		
		# actualizar el tiempo
	t += dt
		
	# calcular la energía del sistema
	KE, PE  = getEnergy( pos, vel, mass, G )
		
	# se guardan las energías en cada instante
	pos_save[:,:,i+1] = pos
	KE_save[i+1] = KE
	PE_save[i+1] = PE
	
	
	# se añaden etiquetas
plt.sca(ax2)
plt.xlabel('time')
plt.ylabel('energy')
ax2.legend(loc='upper right')
plt.show()
	    
def update_plot(frame, pos_save, KE_save, PE_save, t_all, ax1, ax2):
     
     tEnd = 10.0   # tiempo para el cual la simulación termina
     ax1.clear()
     xx = pos_save[:, 0, max(frame-50, 0):frame+1]
     yy = pos_save[:, 1, max(frame-50, 0):frame+1]
     ax1.scatter(xx, yy, s=1, color=[.7, .7, 1])
     ax1.scatter(pos_save[:, 0, frame], pos_save[:, 1, frame], s=10, color='blue')
     ax1.set(xlim=(-2, 2), ylim=(-2, 2))
     ax1.set_aspect('equal', 'box')
     ax1.set_xticks([-2, -1, 0, 1, 2])
     ax1.set_yticks([-2, -1, 0, 1, 2])

     ax2.clear()
     ax2.scatter(t_all[:frame+1], KE_save[:frame+1], color='red', s=1, label='KE')
     ax2.scatter(t_all[:frame+1], PE_save[:frame+1], color='blue', s=1, label='PE')
     ax2.scatter(t_all[:frame+1], KE_save[:frame+1] + PE_save[:frame+1], color='black', s=1, label='Etot')
     ax2.set(xlim=(0, tEnd), ylim=(-300, 300))
     ax2.set_aspect(0.007)
     ax2.legend(loc='upper right')

   
     fig = plt.figure(figsize=(4,5), dpi=80)
     

#%%
#Se corre la animación 

animation = FuncAnimation(fig, update_plot, frames=Nt, fargs=(pos_save, KE_save, PE_save, t_all, ax1, ax2),
         interval=10, repeat=False)
     
animation.save('nbody_animation.mp4', writer='ffmpeg', fps=30)
