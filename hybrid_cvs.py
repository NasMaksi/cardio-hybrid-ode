import torch                 # PyTorch — библиотека для работы с тензорами и нейросетями
from torchdiffeq import odeint  # torchdiffeq — библиотека для решения дифференциальных уравнений с помощью PyTorch
import numpy as np           # numpy — библиотека для работы с массивами и числовыми операциями
import matplotlib.pyplot as plt  # matplotlib — библиотека для визуализации графиков

# Определяем класс нейросети (MLP) с ELU активацией
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Последовательность из линейных слоёв и ELU-активаций
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.ELU(),
            torch.nn.Linear(10, 10),
            torch.nn.ELU(),
            torch.nn.Linear(10, 10),
            torch.nn.ELU(),
            torch.nn.Linear(10, 2)  # На выходе 2 параметра: Pperi и Vspt
        )

    def forward(self, x):
        return self.net(x)

# Класс гибридной модели нейронного ОДУ для сердечно-сосудистой системы
class HybridODEFunc(torch.nn.Module):
    def __init__(self, p_tensor, net):
        super().__init__()
        self.p = p_tensor  # Вектор параметров, загруженный из numpy
        self.net = net     # Нейросеть для предсказания двух величин

    # Метод, вычисляющий производную состояния в момент времени t
    def forward(self, t, u):
        # Проверка состояния на NaN и бесконечности — важна для стабильности
        if torch.isnan(u).any() or torch.isinf(u).any():
            raise RuntimeError(f"NaN/Inf в состоянии при t={t.item()}: {u}")

        # Распаковка переменных состояния (потоки и объемы)
        Qmt, Qav, Qtc, Qpv, Vlv, Vao, Vvc, Vrv, Vpa, Vpu = u
        p = self.p

        # Распаковка параметров модели (те же, что и в Julia)
        (Elvf, Eao, Evc, Ervf, Epa, Epu,
         Rmt, Rav, Rsys, Rtc, Rpv, Rpul,
         Lmt, Lav, Ltc, Lpv,
         Vdlvf, Vdao, Vdvc, Vdrvf, Vdpa, Vdpu,
         P0lvf, P0rvf, lambdalvf, lambdarvf,
         Espt, V0lvf, V0rvf, P0spt, P0pcd,
         V0spt, V0pcd, lambdaspt, lambdapcd,
         Vdspt, Pth) = p

        # Функция временной эластичности e(t), повторяющая фазу сокращения сердца
        e = torch.exp(-80 * ((t % 0.75) - 0.375) ** 2)

        # Вход нейросети — вектор объёмов сердца и сосудов
        inp = torch.stack([Vlv, Vao, Vvc, Vrv, Vpa])
        z = self.net(inp)
        Pperi, Vspt = z[0], z[1]  # Выходы нейросети — дополнительное давление и объём сокращения

        # Коррекция объёмов левого и правого желудочков с учётом Vspt
        Vlvf = Vlv - Vspt
        Vrvf = Vrv + Vspt

        # Вычисляем давление в желудочках с клиппингом экспоненты для численной устойчивости
        exp_lv = torch.exp(torch.clamp(lambdalvf * (Vlvf - V0lvf), max=88.0))
        exp_rv = torch.exp(torch.clamp(lambdarvf * (Vrvf - V0rvf), max=88.0))
        exp_spt = torch.exp(torch.clamp(lambdaspt * (Vspt - V0spt), max=88.0))
        exp_pcd = torch.exp(torch.clamp(lambdapcd * ((Vlv+Vrv) - V0pcd), max=88.0))

        # Давления левого и правого желудочков
        Plvf = e * Elvf * (Vlvf - Vdlvf) + (1 - e) * P0lvf * (exp_lv - 1)
        Prvf = e * Ervf * (Vrvf - Vdrvf) + (1 - e) * P0rvf * (exp_rv - 1)
        Plv = Plvf + Pperi
        Prv = Prvf + Pperi

        # Давления в артериях и венах
        Pao = Eao * (Vao - Vdao)
        Pvc = Evc * (Vvc - Vdvc)
        Ppa = Epa * (Vpa - Vdpa) + Pth
        Ppu = Epu * (Vpu - Vdpu) + Pth

        # Потоки по системному и легочному кругу кровообращения
        Qsys = (Pao - Pvc) / Rsys
        Qpul = (Ppa - Ppu) / Rpul

        # Инициализируем производную состояния
        du = torch.zeros_like(u)

        # Вычисляем производные потоков с учётом сопротивлений и индуктивностей
        du[0] = (Ppu - Plv - Qmt * Rmt) / Lmt if (Ppu - Plv > 0 or Qmt > 0) else 0.0
        du[1] = (Plv - Pao - Qav * Rav) / Lav if (Plv - Pao > 0 or Qav > 0) else 0.0
        du[2] = (Pvc - Prv - Qtc * Rtc) / Ltc if (Pvc - Prv > 0 or Qtc > 0) else 0.0
        du[3] = (Prv - Ppa - Qpv * Rpv) / Lpv if (Prv - Ppa > 0 or Qpv > 0) else 0.0

        # Ограничиваем потоки снизу нулём (клапаны не пропускают обратный ток)
        Qmt = torch.clamp(Qmt, min=0.0)
        Qav = torch.clamp(Qav, min=0.0)
        Qtc = torch.clamp(Qtc, min=0.0)
        Qpv = torch.clamp(Qpv, min=0.0)

        # Вычисляем производные объёмов по балансу потоков
        du[4] = Qmt - Qav
        du[5] = Qav - Qsys
        du[6] = Qsys - Qtc
        du[7] = Qtc - Qpv
        du[8] = Qpv - Qpul
        du[9] = Qpul - Qmt

        # Проверка производных на NaN и бесконечности
        if torch.isnan(du).any() or torch.isinf(du).any():
            raise RuntimeError(f"NaN/Inf в производной при t={t.item()}: {du}")

        return du

def main():
    # Загружаем параметры p как numpy-массив и преобразуем в torch-тензор
    p_np = np.array([
        2.8798, 0.6913, 0.0059, 0.585, 0.369, 0.0073,
        0.0158, 0.018, 1.0889, 0.0237, 0.0055, 0.1552,
        7.6968e-5, 1.2189e-4, 8.0093e-5, 1.4868e-4,
        0, 0, 0, 0, 0, 0, 0.1203, 0.2157, 0.033,
        0.023, 48.754, 0, 0, 1.1101, 0.5003, 2, 200,
        0.435, 0.03, 2, -4
    ], dtype=np.float32)
    p_tensor = torch.from_numpy(p_np)

    net = NeuralNet()
    func = HybridODEFunc(p_tensor, net)

    # Начальное состояние системы (потоки и объёмы)
    u0 = torch.tensor([
        245.5813, 0.0, 190.0661, 0.0, 94.6812,
        133.3381, 329.7803, 90.7302, 43.0123, 808.4579
    ], dtype=torch.float32)

    # Временной интервал интегрирования (0 — 0.3 секунд)
    t = torch.linspace(0.0, 0.3, 30, dtype=torch.float32)

    # Интегрируем систему с помощью torchdiffeq (метод bosh3 для устойчивости)
    try:
        sol = odeint(func, u0, t, method='bosh3', atol=1e-7, rtol=1e-4)
        print("Интегрирование прошло успешно")
    except RuntimeError as e:
        print("Ошибка интегрирования:", e)
        return

    # Визуализация результатов
    sol_np = sol.detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    for i in range(sol_np.shape[1]):
        plt.plot(t.numpy(), sol_np[:, i], label=f'var {i+1}')
    plt.xlabel('Время (с)')
    plt.ylabel('Переменные состояния')
    plt.legend()
    plt.title('Гибридная нейронная модель ОДУ – Сердечно-сосудистая система')
    plt.show()

if __name__ == '__main__':
    main()
