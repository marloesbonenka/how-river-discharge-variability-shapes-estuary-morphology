
#%%
import numpy as np
import matplotlib.pyplot as plt


#%%
Ttide = 12.42 #hours
mean_prism = 5 #m3/s
amplitude = 3

t = np.linspace(0, 24, 24*6  + 1)
y = mean_prism + amplitude * np.sin(2 * np.pi * t / Ttide) 

plt.plot(t, y)
plt.axhline(y=mean_prism, color='r', linestyle='--', label='mean tide')
plt.xlabel('time (hours)')
plt.ylabel('tidal prism (m3/s)')
plt.title('semi-diurnal tide')
plt.legend()
plt.show()
# %%

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

np.random.seed(42)

days = 365
dt = 1 / 24                    # 1-hour time step [days]
time_days = np.arange(0, days, dt)

mean_discharge = 100           # Target mean discharge [m³/s]


# ============================================================
# 1. CONSTANT-ISH HYDROGRAPH
# ============================================================

noise = np.random.normal(0, 20, len(time_days))

window = 24
kernel = np.ones(window) / window
constant = np.convolve(noise, kernel, mode="same")

constant += mean_discharge

# Adjust so the mean is exactly 100 m³/s
constant -= np.mean(constant) - mean_discharge


# ============================================================
# 2. SEASONAL HYDROGRAPH
# ============================================================

seasonal = (
    mean_discharge
    + 50 * np.sin(2 * np.pi * time_days / 365 - np.pi / 2)
)

# Adjust to exact mean
seasonal -= np.mean(seasonal) - mean_discharge


# ============================================================
# 3. PEAKY / FLOOD-DOMINATED HYDROGRAPH
# ============================================================

peaky = np.ones_like(time_days) * 50

n_events = 35

for i in range(n_events):

    event_day = np.random.uniform(0, 365)
    peak = np.random.uniform(100, 500)
    duration = np.random.uniform(0.5, 3.0)

    peaky += peak * np.exp(
        -0.5 * ((time_days - event_day) / duration) ** 2
    )

# Scale to target mean
peaky *= mean_discharge / np.mean(peaky)


# ============================================================
# CALCULATE MEANS
# ============================================================

mean_constant = np.mean(constant)
mean_seasonal = np.mean(seasonal)
mean_peaky = np.mean(peaky)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(8, 9),
    sharex=True
)


# ------------------------------------------------------------
# 1. CONSTANT
# ------------------------------------------------------------

axes[0].plot(
    time_days,
    constant,
    linewidth=1.2,
    color='tab:blue'
)

axes[0].axhline(
    mean_constant,
    linestyle="--",
    linewidth=1.5,
    color='tab:blue'
)

# Direct mean label
axes[0].text(
    350,
    mean_constant + 12,
    f"mean = {mean_constant:.1f} m³/s",
    color='tab:blue',
    ha='right',
    va='bottom'
)

axes[0].set_ylabel("discharge [m³/s]")
axes[0].set_title("constant river discharge")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 450)


# ------------------------------------------------------------
# 2. SEASONAL
# ------------------------------------------------------------

axes[1].plot(
    time_days,
    seasonal,
    linewidth=1.5,
    color='tab:orange'
)

axes[1].axhline(
    mean_seasonal,
    linestyle="--",
    linewidth=1.5,
    color='tab:orange'
)

# Direct mean label
axes[1].text(
    350,
    mean_seasonal + 12,
    f"mean = {mean_seasonal:.1f} m³/s",
    color='tab:orange',
    ha='right',
    va='bottom'
)

axes[1].set_ylabel("discharge [m³/s]")
axes[1].set_title("seasonal river discharge")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 450)


# ------------------------------------------------------------
# 3. PEAKY
# ------------------------------------------------------------

axes[2].plot(
    time_days,
    peaky,
    linewidth=1.0,
    color='tab:green'
)

axes[2].axhline(
    mean_peaky,
    linestyle="--",
    linewidth=1.5,
    color='tab:green'
)

# Direct mean label
axes[2].text(
    350,
    mean_peaky + 12,
    f"mean = {mean_peaky:.1f} m³/s",
    color='tab:green',
    ha='right',
    va='bottom'
)

axes[2].set_ylabel("discharge [m³/s]")
axes[2].set_xlabel("Time [days]")
axes[2].set_title("flood-dominated river discharge")
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 450)


# ============================================================
# FINAL FORMATTING
# ============================================================

plt.tight_layout()
plt.show()


# ============================================================
# PRINT MEANS
# ============================================================

print(f"Mean constant discharge: {mean_constant:.2f} m³/s")
print(f"Mean seasonal discharge: {mean_seasonal:.2f} m³/s")
print(f"Mean peaky discharge:    {mean_peaky:.2f} m³/s")

# %%
