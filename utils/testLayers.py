import numpy as np

from examples.seismic import Model, plot_velocity

# Define a physical size
shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # What is the location of the top left corner. This is necessary to define
# the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 2.5

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=2, nbl=10, bcs="damp")


from examples.seismic import TimeAxis

t0 = 0.  # Simulation starts a t=0
tn = 1000.  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)

from examples.seismic import RickerSource

f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = 0.  # Depth is 20m


#NBVAL_IGNORE_OUTPUT
from examples.seismic import Receiver

# Create symbol for 101 receivers
rec = Receiver(name='rec', grid=model.grid, npoint=101, time_range=time_range)

# Prescribe even spacing for receivers along the x-axis
rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
rec.coordinates.data[:, 1] = 20.  # Depth is 20m

# We can now show the source and receivers within our domain:
# Red dot: Source location
# Green dots: Receiver locations (every 4th point)
plot_velocity(model, source=src.coordinates.data,
              receiver=rec.coordinates.data[::4, :])


from devito import TimeFunction

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)

# We can now write the PDE
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt


from devito import Eq, solve

stencil = Eq(u.forward, solve(pde, u.forward))
#src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
src_term = src.inject(field=u.forward, expr=src)

# Create interpolation expression for receivers
rec_term = rec.interpolate(expr=u.forward)

from devito import Operator

op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

op(time=time_range.num-1, dt=model.critical_dt)
from examples.seismic import plot_shotrecord

plot_shotrecord(rec.data, model, t0, tn)
np.set_printoptions(threshold=np.inf)

import pandas as pd

df = pd.DataFrame(rec.data) #Reciever data
nreceivers = 101

import os

# Crear carpeta de salida
output_folder = "LayerSim"
os.makedirs(output_folder, exist_ok=True)

# === 1. Exportar señal del pulso de Ricker ===
ricker_df = pd.DataFrame({
    "time (ms)": src.time_values,  
    "amplitude": src.data[:,0]
})
ricker_df.to_csv(os.path.join(output_folder, "ricker_layers.csv"), index=False)


# === 2. Exportar señal de cada recibidor ===

# Tiempo en milisegundos

df = pd.DataFrame(rec.data)

# Crear DataFram

receiver_data = pd.DataFrame(rec.data, columns=[f'Receiver {i}' for i in range(nreceivers)])
receiver_data.to_csv(os.path.join(output_folder, "recData.csv"), index=False)


receiver_data.insert(0, 'Time (ms)', src.time_values)

geofonos_df = pd.DataFrame(receiver_data)
geofonos_df.to_csv(os.path.join(output_folder, "geofonos_layer.csv"), index=False)
