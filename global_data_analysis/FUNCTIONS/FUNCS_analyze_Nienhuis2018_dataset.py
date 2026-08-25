import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#  --- FUNCTIONS for analyzing estuaries from the Nienhuis 2018 dataset ---

def load_data(filepath: Path) -> pd.DataFrame:
    """Load estuary data from Excel, dropping fully empty rows."""
    df = pd.read_excel(filepath, header=0, skiprows=[1])
    return df.dropna(how='all')


def select_estuaries(df: pd.DataFrame, 
                     select_T: bool = True, t_lower: float = 1, t_upper: float = 100,
                     select_Q: bool = True, q_upper: float = 1000,
                     drop_outliers: list = None) -> pd.DataFrame:
    """Filter estuaries based on tidal range (T) and discharge (Q) criteria."""
    df = df.set_index('Name')

    if select_T:
        df = df[(df['T'] > t_lower) & (df['T'] < t_upper)]
        print(f"Estuaries after T selection ({t_lower} < T < {t_upper}): {len(df)}")

    if drop_outliers:
        df = df.drop(labels=drop_outliers, errors='ignore')
        print(f"Dropped outliers: {drop_outliers}")

    if select_Q:
        df = df[df['Qriver (m3s-1)'] < q_upper]
        print(f"Estuaries after Q selection (Q < {q_upper}): {len(df)}")

    return df


def plot_discharge_pdf(df: pd.DataFrame, title: str, save_path: Path = None):
    """Plot and optionally save a probability density histogram of river discharge."""
    plt.figure()
    df['Qriver (m3s-1)'].plot(kind='hist', bins=30, density=True, alpha=0.7)
    plt.xlabel('River Discharge (m³/s)')
    plt.ylabel('Probability Density')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_discharge_kde(df: pd.DataFrame, title: str, save_path: Path = None):
    """Plot a KDE of river discharge."""
    plt.figure()
    sns.histplot(df['Qriver (m3s-1)'], kde=True, stat='density', bins=8, alpha=0.5)
    plt.xlabel('River Discharge (m³/s)')
    plt.ylabel('Probability Density')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_discharge_bar(df: pd.DataFrame, title: str, save_path: Path = None):
    """Plot and optionally save a bar chart of discharge per estuary."""
    ax = df['Qriver (m3s-1)'].plot(kind='bar')
    ax.set_xlabel('Estuary')
    ax.set_ylabel('River Discharge (m³/s)')
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_selection(df: pd.DataFrame, output_path: Path):
    """Save the selected estuary DataFrame to an Excel file."""
    df.to_excel(output_path)
    print(f"Saved selection to {output_path}")