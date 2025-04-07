import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import pandas as pd
import torch
import torch.nn as nn

# Configurar dispositivo CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

###############################################################################
# 1. Clases y funciones base
###############################################################################

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function):
        super(MLP, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_units)
        self.linear_out = nn.Linear(hidden_units, output_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers)])
        self.act = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x

def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for _ in range(order):
        dy = torch.autograd.grad(
            dy, x,
            grad_outputs=torch.ones_like(dy),
            create_graph=True,
            retain_graph=True
        )[0]
    return dy

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

###############################################################################
# 2. Generar datos (observados y de colocalización)
###############################################################################

def generate_interior_data(path="receiver_fourier.csv"):
    # Leer el CSV
    df = pd.read_csv(path)

    x_data = df[['Position X']].values
    y_data = df[['Position Y']].values
    u_data_np = df[['Real']].values  # u: valor de la solución real
    u_data_imag_np = df[['Imag']].values  # u: valor de la solución real


    # Concatenar x, y para formar X_data_np
    X_data_np = np.hstack((x_data, y_data))

    # Convertir a tensores de PyTorch con el mismo formato que en el original
    X_data = torch.tensor(X_data_np, dtype=torch.float32, requires_grad=True, device=device)
    u_data = torch.tensor(u_data_np, dtype=torch.float32, device=device)
    u_data_imag = torch.tensor(u_data_imag_np, dtype=torch.float32, device=device)

    return X_data, u_data, u_data_imag

def generate_collocation_points(N_interior=1500, N_neumann = 200, N_boundary=200, std_dev=0.2, csv_path='ricker_fourier.csv'):
    """
    Genera puntos de colocalización (interior y frontera) para entrenar la parte PDE y BC.
    """
    x_int = np.random.normal(loc=0.5, scale=0.35, size=(N_interior, 1))
    y_int = -np.random.exponential(scale=0.35, size=(N_interior, 1))

    # Asegurar que los puntos estén dentro del dominio [-0.1,1.1] x [-1,0]
    #x_int = np.clip(x_int, 0, 1)
    #y_int = np.clip(y_int, -1, 0)

    X_int_np = np.hstack((x_int, y_int))
    X_int = torch.tensor(X_int_np, dtype=torch.float32, requires_grad=True,device=device)

    # Leer los datos de frontera desde el CSV
    df = pd.read_csv(csv_path)

    # Limitar al número de puntos de frontera que queremos
    magnitudes = df["Real"].values[:N_boundary].reshape(-1, 1)
    magnitudesIm = df["Imag"].values[:N_boundary].reshape(-1, 1)


    # Crear coordenadas fijas de frontera: x = 0.5, y = 0
    x_bnd = np.full_like(magnitudes, 0.5)
    y_bnd = np.zeros_like(magnitudes)
    X_bnd_np = np.hstack((x_bnd, y_bnd))

    # Crear coordenadas fijas de frontera: x = 0.5, y = 0
    x_bndIm = np.full_like(magnitudesIm, 0.5)
    y_bndIm = np.zeros_like(magnitudesIm)
    X_bnd_npIm = np.hstack((x_bndIm, y_bndIm))

    X_bnd = torch.tensor(X_bnd_np, dtype=torch.float32, requires_grad=True, device=device)
    u_bnd = torch.tensor(magnitudes, dtype=torch.float32, device=device)

    X_bndIm = torch.tensor(X_bnd_npIm, dtype=torch.float32, requires_grad=True, device=device)
    u_bndIm = torch.tensor(magnitudesIm, dtype=torch.float32, device=device)

    x_neu = np.linspace(0, 1, N_neumann).reshape(-1, 1)
    x_neu = x_neu[x_neu[:, 0] != 0.5]  # excluimos el punto 0.5 para no duplicar
    y_neu = np.zeros_like(x_neu)
    X_neu_np = np.hstack((x_neu, y_neu))
    X_neumann = torch.tensor(X_neu_np, dtype=torch.float32, requires_grad=True, device=device)


    return X_int, X_bnd, u_bnd, X_neumann, X_bndIm, u_bndIm
###############################################################################
# 3. Definición de las pérdidas para el problema inverso
###############################################################################

def loss_pde_inverse(model, X_int):
    output = model(X_int)
    u = output[:, 0:1]
    c = output[:, 1:2]
    u_im = output[:, 2:3]

    grads_u = derivative(u, X_int, order=1)
    u_x = grads_u[:, 0:1]
    u_y = grads_u[:, 1:2]

    grads_uIm = derivative(u_im, X_int , order=1)
    ui_x = grads_uIm[:, 0:1]
    ui_y = grads_uIm[:, 1:2]

    u_xx = derivative(u_x, X_int, order=1)[:, 0:1]
    u_yy = derivative(u_y, X_int, order=1)[:, 1:2]

    ui_xx = derivative(ui_x, X_int, order=1)[:, 0:1]
    ui_yy = derivative(ui_y, X_int, order=1)[:, 1:2]

    forcing = 0
    omega = 0.2

    residual = (omega**2/c**2)*(u+u_im) + (u_xx + u_yy + ui_xx + ui_yy) - forcing
    return torch.mean(residual**2)

def loss_bc(model, X_bnd, u_realbnd):
    u_b = model(X_bnd)[:, 0:1]
    return torch.mean((u_b-u_realbnd)**2)

def loss_bcIm(model, X_bnd, u_realbnd):
    u_b = model(X_bnd)[:, 2:3]
    return torch.mean((u_b-u_realbnd)**2)

def loss_data(model, X_data, u_data):
    u_pred = model(X_data)[:, 0:1]
    return torch.mean((u_pred - u_data)**2)

def loss_data_im(model, X_data, u_data):
    u_pred = model(X_data)[:, 2:3]
    return torch.mean((u_pred - u_data)**2)

def loss_neumann(model, X_neumann):
    u = model(X_neumann)[:, 0:1]
    grads = derivative(u, X_neumann, order=1)
    u_y = grads[:, 1:2]  # dy is the second column
    return torch.mean(u_y**2)

def loss_neumann_im(model, X_neumann):
    u = model(X_neumann)[:, 2:3]
    grads = derivative(u, X_neumann, order=1)
    u_y = grads[:, 1:2]  # dy is the second column
    return torch.mean(u_y**2)

###############################################################################
# 4. Función para graficar la solución y k
###############################################################################

def plot_solution_and_k(model, epoch, folder="figs_inverse_mixed", n_points=150):
    if not os.path.exists(folder):
        os.makedirs(folder)

    x_vals = np.linspace(0, 1, n_points)
    y_vals = np.linspace(-1, 0, n_points)
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    XY_np = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T
    XY_torch = torch.tensor(XY_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        output = model(XY_torch)
        u_pred = output[:, 0].cpu().numpy().reshape(n_points, n_points)
        k_pred = output[:, 1].cpu().numpy().reshape(n_points, n_points)
        uim_pred = output[:, 2].cpu().numpy().reshape(n_points, n_points)
    fig = plt.figure(figsize=(18, 5))  # wider figure for 3 plots

    # First subplot — u
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.pcolormesh(X_mesh, Y_mesh, u_pred, cmap='viridis', shading='auto')
    ax1.set_title(f"PINN Re(u) (epoch {epoch})")
    fig.colorbar(im1, ax=ax1)

    # Second subplot — k
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.pcolormesh(X_mesh, Y_mesh, uim_pred, cmap='viridis', shading='auto')
    ax2.set_title(f"PINN Im(u) (epoch {epoch})")
    fig.colorbar(im2, ax=ax2)

    # Third subplot — w
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.pcolormesh(X_mesh, Y_mesh, k_pred, cmap='viridis', shading='auto')
    ax3.set_title(f"PINN k (epoch {epoch})")
    fig.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"solution_epoch_{epoch}.png"))
    plt.close(fig)



###############################################################################
# 5. Entrenamiento
###############################################################################

def train_inverse_pinn_mixed(
    model, 
    X_int, X_bnd, u_bnd,
    X_data, u_data, x_neu, u_dataim, x_bndim, u_bndim,
    adam_epochs=10000,
    lbfgs_iterations=500,
    lr_adam=1e-4,
    lr_lbfgs=0.5,
    lambda_bc=5.0, 
    lambda_data=1.0,
    plot_every=1000
):
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)

    print(">>> FASE 1: Entrenamiento con Adam <<<")
    for epoch in tqdm(range(1, adam_epochs+1), desc="Adam"):
        optimizer_adam.zero_grad()

        pde_loss = loss_pde_inverse(model, X_int)
        bc_loss_val = loss_bc(model, X_bnd,u_bnd)
        bc_loss_valIm = loss_bcIm(model, x_bndim, u_bndim)
        data_loss_val = loss_data(model, X_data, u_data)
        data_loss_im = loss_data_im(model, X_data,u_dataim)
        neumann_loss = loss_neumann(model,x_neu)
        neumann_loss_im = loss_neumann_im(model, x_neu)

        total_loss = pde_loss + lambda_bc * bc_loss_val + lambda_data * data_loss_val + neumann_loss + data_loss_im + bc_loss_valIm + neumann_loss_im
        total_loss.backward()
        optimizer_adam.step()

        if epoch % plot_every == 0 or epoch == 1 or epoch == adam_epochs:
            print(f"  [Adam epoch {epoch:5d}] total_loss={total_loss.item():.4e}, "
                  f"pde_loss={pde_loss.item():.4e}, bc_loss={bc_loss_val.item():.4e}, "
                  f"data_loss={data_loss_val.item():.4e},"
                  f"neumann_loss={neumann_loss.item():.4e},"
                  f"dataIm_loss={data_loss_im.item():.4e},"
                  f"boundaryIm_loss={bc_loss_valIm.item():.4e},"
                  f"NeumannImLoss={neumann_loss_im.item():.4e},")
            plot_solution_and_k(model, epoch, folder="figs_inverse_mixed_adam")

    print(">>> FASE 2: Entrenamiento con L-BFGS <<<")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lr_lbfgs,
        max_iter=lbfgs_iterations,
        history_size=100
    )

    iteration_lbfgs = [0]
    def closure():
        optimizer_lbfgs.zero_grad()
        pde_loss = loss_pde_inverse(model, X_int)
        bc_loss_val = loss_bc(model, X_bnd,u_bnd)
        data_loss_val = loss_data(model, X_data, u_data)
        data_loss_im = loss_data_im(model, X_data,u_dataim)
        bc_loss_valIm = loss_bcIm(model, x_bndim, u_bndim)
        neumann_loss = loss_neumann(model,x_neu)
        neumann_loss_im = loss_neumann_im(model, x_neu)


        total_loss = pde_loss + lambda_bc * bc_loss_val + lambda_data * data_loss_val + neumann_loss + data_loss_im + bc_loss_valIm + neumann_loss_im
        total_loss.backward()
        return total_loss

    for i in tqdm(range(1, lbfgs_iterations+1)):
        loss_value = optimizer_lbfgs.step(closure)
        iteration_lbfgs[0] += 1
        if (i+1) % 100 == 0 or (i+1) == lbfgs_iterations:
            current_pde = loss_pde_inverse(model, X_int).item()
            current_bc = loss_bc(model, X_bnd,u_bnd).item()
            current_data = loss_data(model, X_data, u_data).item()
            current_total = current_pde + lambda_bc * current_bc + lambda_data * current_data 

            print(f"  [Adam epoch {epoch:5d}] total_loss={total_loss.item():.4e}, "
                  f"pde_loss={pde_loss.item():.4e}, bc_loss={bc_loss_val.item():.4e}, "
                  f"data_loss={data_loss_val.item():.4e},"
                  f"neumann_loss={neumann_loss.item():.4e},"
                  f"dataIm_loss={data_loss_im.item():.4e},"
                  f"boundaryIm_loss={bc_loss_valIm.item():.4e},")

    return model

###############################################################################
# 6. Ejecución principal
###############################################################################

if __name__ == "__main__":
    set_seed(42)

    X_int, X_bnd, u_bnd, X_neu, x_bndIm, u_bndIm = generate_collocation_points(N_interior=25000, N_boundary=100, N_neumann=500, csv_path="ricker_fourier.csv")
    X_data, u_data, u_dataIm = generate_interior_data(path="receiver_fourier.csv")
    

    model = MLP(
        input_size=2,
        output_size=3,
        hidden_layers=32,
        hidden_units=64,
        activation_function=nn.Tanh()
    ).to(device)
    model.apply(init_weights)

    # Inicialización personalizada para k (función lineal de y)
    with torch.no_grad():
        # Forzar que k = 1 - y inicialmente
        # Capa de entrada: pasar y a una neurona oculta
        model.linear_in.weight.data[0, 1] = 1.0  # Conexión fuerte a y
        model.linear_in.bias.data[0] = 0.0
        
        # Capas ocultas: propagar y sin alteraciones (activación ~lineal en [-1,0])
        for layer in model.layers:
            layer.weight.data[:, :] *= 0.01  # Reducir influencia de otras neuronas
            layer.weight.data[0, 0] = 1.0    # Mantener la neurona de y
            layer.bias.data[:] = 0.0
        
        # Capa de salida: k = 1 - y  
        model.linear_out.weight.data[1, :] = 0.0  # Desconectar otras neuronas
        model.linear_out.weight.data[1, 0] = -1.0  # Peso para la neurona de y
        model.linear_out.bias.data[1] = 1.0        # Bias para k=1 en y=0

    model = train_inverse_pinn_mixed(
        model,
        X_int, X_bnd, u_bnd,
        X_data, u_data, X_neu, u_dataIm, x_bndIm, u_bndIm,
        adam_epochs=10000,
        lbfgs_iterations=500,
        lr_adam=1e-4,
        lr_lbfgs=0.1,
        lambda_bc=10.0,
        lambda_data=10.0,
        plot_every=500)

    plot_solution_and_k(model, 0, folder="figs_inverse_mixed_final")