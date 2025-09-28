import torch
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

class ReservoirNet(torch.nn.Module):
    """
    - Input: 5 state volumes
    - Reservoir: random projection + nonlinearity
    - Readout: linear layer mapping to 2 outputs (Pperi, Vspt)
    """
    def __init__(self, input_dim=5, reservoir_size=50, output_dim=2):
        super().__init__()
        # Fixed random reservoir weights (like RC)
        self.W_in = torch.nn.Parameter(torch.randn(input_dim, reservoir_size)*0.1,
                                       requires_grad=False)
        self.W_res = torch.nn.Parameter(torch.randn(reservoir_size, reservoir_size)*0.05,
                                        requires_grad=False)
        # Trainable readout (like NG-RC + MLP readout)
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(reservoir_size, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, output_dim)
        )

    def forward(self, x):
        # Project input to reservoir space
        r = torch.tanh(x @ self.W_in + torch.matmul(x @ self.W_in, self.W_res))
        return self.readout(r)

# -----------------------------
# Hybrid ODE Function
# -----------------------------
class HybridODEFunc(torch.nn.Module):
    def __init__(self, p_tensor, net):
        super().__init__()
        self.p = p_tensor
        self.net = net

    def forward(self, t, u):
        Qmt, Qav, Qtc, Qpv, Vlv, Vao, Vvc, Vrv, Vpa, Vpu = u
        p = self.p

        (Elvf, Eao, Evc, Ervf, Epa, Epu,
         Rmt, Rav, Rsys, Rtc, Rpv, Rpul,
         Lmt, Lav, Ltc, Lpv,
         Vdlvf, Vdao, Vdvc, Vdrvf, Vdpa, Vdpu,
         P0lvf, P0rvf, lambdalvf, lambdarvf,
         Espt, V0lvf, V0rvf, P0spt, P0pcd,
         V0spt, V0pcd, lambdaspt, lambdapcd,
         Vdspt, Pth) = p

        # Heart activation function
        e = torch.exp(-80 * ((t % 0.75) - 0.375) ** 2)

        # NN input = volumes
        inp = torch.stack([Vlv, Vao, Vvc, Vrv, Vpa]).unsqueeze(0)
        z = self.net(inp).squeeze(0)
        Pperi, Vspt = z[0], z[1]

        # Adjusted ventricular volumes
        Vlvf = Vlv - Vspt
        Vrvf = Vrv + Vspt

        # Exponentials (safe with clamp)
        exp_lv = torch.exp(torch.clamp(lambdalvf * (Vlvf - V0lvf), max=88.0))
        exp_rv = torch.exp(torch.clamp(lambdarvf * (Vrvf - V0rvf), max=88.0))

        # Pressures
        Plvf = e * Elvf * (Vlvf - Vdlvf) + (1 - e) * P0lvf * (exp_lv - 1)
        Prvf = e * Ervf * (Vrvf - Vdrvf) + (1 - e) * P0rvf * (exp_rv - 1)
        Plv, Prv = Plvf + Pperi, Prvf + Pperi
        Pao, Pvc = Eao * (Vao - Vdao), Evc * (Vvc - Vdvc)
        Ppa, Ppu = Epa * (Vpa - Vdpa) + Pth, Epu * (Vpu - Vdpu) + Pth

        Qsys = (Pao - Pvc) / Rsys
        Qpul = (Ppa - Ppu) / Rpul

        du = torch.zeros_like(u)
        du[0] = (Ppu - Plv - Qmt * Rmt) / Lmt if (Ppu - Plv > 0 or Qmt > 0) else 0.0
        du[1] = (Plv - Pao - Qav * Rav) / Lav if (Plv - Pao > 0 or Qav > 0) else 0.0
        du[2] = (Pvc - Prv - Qtc * Rtc) / Ltc if (Pvc - Prv > 0 or Qtc > 0) else 0.0
        du[3] = (Prv - Ppa - Qpv * Rpv) / Lpv if (Prv - Ppa > 0 or Qpv > 0) else 0.0

        Qmt, Qav, Qtc, Qpv = map(lambda q: torch.clamp(q, min=0.0), [Qmt, Qav, Qtc, Qpv])

        du[4] = Qmt - Qav
        du[5] = Qav - Qsys
        du[6] = Qsys - Qtc
        du[7] = Qtc - Qpv
        du[8] = Qpv - Qpul
        du[9] = Qpul - Qmt

        return du
        
# Main program with user input
def main():
    # Parameters
    p_np = np.array([
        2.8798, 0.6913, 0.0059, 0.585, 0.369, 0.0073,
        0.0158, 0.018, 1.0889, 0.0237, 0.0055, 0.1552,
        7.6968e-5, 1.2189e-4, 8.0093e-5, 1.4868e-4,
        0, 0, 0, 0, 0, 0, 0.1203, 0.2157, 0.033,
        0.023, 48.754, 0, 0, 1.1101, 0.5003, 2, 200,
        0.435, 0.03, 2, -4
    ], dtype=np.float32)
    p_tensor = torch.from_numpy(p_np)

    # User input for initial condition
    print("Введите начальные условия (10 значений через пробел, иначе будет дефолт):")
    try:
        u0_values = list(map(float, input().split()))
        if len(u0_values) != 10:
            raise ValueError
    except:
        print("Использую дефолтные значения.")
        u0_values = [245.5813, 0.0, 190.0661, 0.0, 94.6812,
                     133.3381, 329.7803, 90.7302, 43.0123, 808.4579]

    u0 = torch.tensor(u0_values, dtype=torch.float32)

    # User input for duration
    try:
        duration = float(input("Введите длительность моделирования (сек): "))
    except:
        duration = 0.3
        print("Дефолт: 0.3 сек")

    t = torch.linspace(0.0, duration, 200, dtype=torch.float32)

    # Solve ODE
    net = ReservoirNet()
    func = HybridODEFunc(p_tensor, net)
    sol = odeint(func, u0, t, method='bosh3')

    # Plot results
    sol_np = sol.detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    for i in range(sol_np.shape[1]):
        plt.plot(t.numpy(), sol_np[:, i], label=f'var{i+1}')
    plt.xlabel('Время (с)')
    plt.ylabel('Значения переменных состояния')
    plt.legend()
    plt.title('Гибридная ODE модель сердечно-сосудистой системы')
    plt.show()

if __name__ == '__main__':
    main()
