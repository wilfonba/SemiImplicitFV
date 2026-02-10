import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("explicitError.csv", delimiter=",", skip_header=1)
res = data[:, 0]
L1E = data[:, 1]
L2E = data[:, 2]
LinfE = data[:, 3]

data = np.genfromtxt("semiImplicitError.csv", delimiter=",", skip_header=1)
L1SI = data[:, 1]
L2SI = data[:, 2]
LinfSI = data[:, 3]

h = res

fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
ax[0].plot(res, L1E, "o-", label="L1", color="red", linewidth=2)
ax[0].loglog(h, 4*L1E[0] * (h / h[0]) ** -5, "-", color="black", label="5th order")
ax[0].plot(res, L1SI, "o-", label="L1 Semi-Implicit", color="blue", linewidth=2)
ax[0].loglog(h, 2*L1SI[0] * (h / h[0]) ** -3, "--", color="black", label="3rd order")
ax[0].set_yscale("log")
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("N")
ax[0].set_ylabel(r"$||e||_1$")
ax[0].set_title(r"$L_{1}$ Error Comparison")

ax[1].plot(res, L2E, "o-", label="L2", color="red", linewidth=2)
ax[1].loglog(h, 4*L2E[0] * (h / h[0]) ** -5, "-", color="black", label="5th order")
ax[1].plot(res, L2SI, "o-", label="L2 Semi-Implicit", color="blue", linewidth=2)
ax[1].loglog(h, 2*L2SI[0] * (h / h[0]) ** -3, "--", color="black", label="3rd order")
ax[1].set_yscale("log")
ax[1].set_xscale("log", base=2)
ax[1].set_xlabel("N")
ax[1].set_ylabel(r"$||e||_2$")
ax[1].set_title(r"$L_{2}$ Error Comparison")

ax[2].plot(res, LinfE, "o-", label="Explicit (CFL = 0.8)", color="red", linewidth=2)
ax[2].loglog(h, 4*LinfE[0] * (h / h[0]) ** -5, "-", color="black", label="5th order")
ax[2].plot(res, LinfSI, "o-", label=r"Semi-Implicit (CFL $\approx$ 27)", color="blue", linewidth=2)
ax[2].loglog(h, 2*LinfSI[0] * (h / h[0]) ** -3, "--", color="black", label="3rd order")
ax[2].set_yscale("log")
ax[2].set_xscale("log", base=2)
ax[2].set_xlabel("N")
ax[2].set_ylabel(r"$||e||_{inf}$")
ax[2].set_title(r"$L_{\infty}$ Error Comparison")

ax[2].legend(loc="lower left")


fig.tight_layout()
# plt.savefig("explicitError.png", dpi=150)
plt.show()
