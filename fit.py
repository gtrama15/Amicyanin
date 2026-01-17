import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STRETCHED EXPONENTIAL FUNCTION (KWW)
# ============================================
def stretched_exponential(t, tau, beta, A):
    """
    Stretched exponential function (Kohlrausch-Williams-Watts)
    
    Parameters:
    -----------
    t : time array
    tau : characteristic relaxation time
    beta : stretching exponent (0 < beta <= 1)
    A : amplitude (should be ~1 for correlation functions)
    
    Returns:
    --------
    C(t) = A * exp(-(t/tau)^beta)
    """
    return A * np.exp(-(t/tau)**beta)

# ============================================
# LOAD GROMACS DATA
# ============================================
def load_gromacs_hbond_correlation(filename):
    """
    Load hydrogen bond lifetime correlation data from GROMACS output.
    Usually from 'hbond.xvg' or similar file.
    
    GROMACS xvg files typically have:
    - First column: time (ps or ns)
    - Second column: correlation function C(t)
    - May have comments starting with # or @
    """
    # Read data, skipping comment lines
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if line and not line.startswith(('#', '@')):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        time = float(parts[0])
                        corr = float(parts[1])
                        data.append([time, corr])
                    except ValueError:
                        continue
    
    data = np.array(data)
    times = data[:, 0]
    correlations = data[:, 1]
    
    print(f"Loaded {len(times)} data points")
    print(f"Time range: {times[0]:.2f} to {times[-1]:.2f} ps")
    print(f"Correlation range: {correlations[0]:.4f} to {correlations[-1]:.4f}")
    
    return times, correlations

# ============================================
# FITTING FUNCTION WITH ERROR HANDLING
# ============================================
def fit_stretched_exponential(times, correlations, initial_guess=None):
    """
    Fit stretched exponential to correlation data.
    
    Parameters:
    -----------
    times : array of time values
    correlations : array of correlation values
    initial_guess : initial parameters [tau, beta, A]
    
    Returns:
    --------
    popt : optimal parameters [tau, beta, A]
    pcov : covariance matrix
    fitted_curve : fitted values
    """
    
    # Default initial guess if not provided
    if initial_guess is None:
        # Estimate initial parameters
        time_range = times[-1] - times[0]
        tau_guess = time_range / 10  # Start with 10% of time range
        beta_guess = 0.7  # Typical for many systems
        A_guess = correlations[0] if correlations[0] > 0 else 1.0
        initial_guess = [tau_guess, beta_guess, A_guess]
    
    print(f"Initial guess: tau={initial_guess[0]:.2f}, "
          f"beta={initial_guess[1]:.2f}, A={initial_guess[2]:.2f}")
    
    # Define bounds (0 < beta <= 1, tau > 0, A > 0)
    bounds = ([0.01, 0.01, 0.01], [np.inf, 1.0, np.inf])
    
    try:
        # Perform the fit
        popt, pcov = curve_fit(
            stretched_exponential, 
            times, 
            correlations, 
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000  # Increase max iterations if needed
        )
        
        # Calculate fitted curve
        fitted_curve = stretched_exponential(times, *popt)
        
        # Calculate R-squared
        residuals = correlations - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((correlations - np.mean(correlations))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard errors from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        
        print("\n" + "="*50)
        print("FITTING RESULTS")
        print("="*50)
        print(f"τ (relaxation time) = {popt[0]:.4f} ± {perr[0]:.4f} ps")
        print(f"β (stretching exponent) = {popt[1]:.4f} ± {perr[1]:.4f}")
        print(f"A (amplitude) = {popt[2]:.4f} ± {perr[2]:.4f}")
        print(f"R² = {r_squared:.6f}")
        print("="*50)
        
        # Calculate half-life (time when C(t) = 0.5 * A)
        t_half = popt[0] * (-np.log(0.5))**(1/popt[1])
        print(f"Half-life (C(t) = 0.5*A): {t_half:.4f} ps")
        
        return popt, pcov, fitted_curve, r_squared
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, None, None

# ============================================
# PLOTTING FUNCTION
# ============================================
def plot_fit_results(times, correlations, fitted_curve, popt, 
                     title="Hydrogen Bond Lifetime Correlation"):
    """
    Plot original data and fitted curve.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Linear scale
    ax1.scatter(times, correlations, alpha=0.6, label='Data', color='blue', s=10)
    ax1.plot(times, fitted_curve, 'r-', linewidth=2, label='Stretched Exponential Fit')
    ax1.set_xlabel('Time (ps)', fontsize=12)
    ax1.set_ylabel('C(t)', fontsize=12)
    ax1.set_title(f'{title} - Linear Scale', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add fitting parameters as text
    fit_text = f'τ = {popt[0]:.2f} ps\nβ = {popt[1]:.3f}\nA = {popt[2]:.3f}'
    ax1.text(0.05, 0.05, fit_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Log-log scale
    ax2.scatter(times, correlations, alpha=0.6, label='Data', color='blue', s=10)
    ax2.plot(times, fitted_curve, 'r-', linewidth=2, label='Fit')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (ps, log scale)', fontsize=12)
    ax2.set_ylabel('C(t), log scale', fontsize=12)
    ax2.set_title(f'{title} - Log-Log Scale', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# ============================================
# RESIDUALS ANALYSIS
# ============================================
def plot_residuals(times, correlations, fitted_curve):
    """
    Plot residuals to check quality of fit.
    """
    residuals = correlations - fitted_curve
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs time
    ax1.scatter(times, residuals, alpha=0.6, s=10)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Time')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # ============================================
    # 1. LOAD YOUR DATA
    # ============================================
    # Replace with your actual GROMACS output file
    filename = "hbond_correlation.xvg"  # Your GROMACS file
    
    try:
        times, correlations = load_gromacs_hbond_correlation(filename)
    except FileNotFoundError:
        print(f"File {filename} not found. Creating example data...")
        # Create example data for demonstration
        times = np.logspace(-2, 3, 200)  # 0.01 to 1000 ps
        true_tau = 50.0  # ps
        true_beta = 0.6
        true_A = 1.0
        correlations = stretched_exponential(times, true_tau, true_beta, true_A)
        # Add some noise
        np.random.seed(42)
        correlations += np.random.normal(0, 0.02, len(correlations))
        correlations = np.clip(correlations, 0, None)
    
    # ============================================
    # 2. OPTIONAL: TRUNCATE DATA
    # ============================================
    # Sometimes early or late points are noisy
    # You might want to exclude very short or very long times
    mask = (times > 0.1) & (times < times[-1]*0.95)  # Adjust as needed
    times_fit = times[mask]
    corr_fit = correlations[mask]
    
    print(f"\nUsing {len(times_fit)} points for fitting")
    
    # ============================================
    # 3. PERFORM FITTING
    # ============================================
    # You can provide initial guess if you have estimates
    initial_guess = [100, 0.7, 1.0]  # [tau, beta, A]
    
    result = fit_stretched_exponential(times_fit, corr_fit, initial_guess)
    
    if result[0] is not None:
        popt, pcov, fitted_curve, r_squared = result
        
        # ============================================
        # 4. PLOT RESULTS
        # ============================================
        fig1 = plot_fit_results(times, correlations, 
                                stretched_exponential(times, *popt), 
                                popt,
                                title="Hydrogen Bond Lifetime Correlation")
        
        fig2 = plot_residuals(times_fit, corr_fit, fitted_curve)
        
        plt.show()
        
        # ============================================
        # 5. SAVE RESULTS
        # ============================================
        # Save fitting parameters
        results_dict = {
            'tau_ps': popt[0],
            'tau_error': np.sqrt(pcov[0, 0]),
            'beta': popt[1],
            'beta_error': np.sqrt(pcov[1, 1]),
            'amplitude': popt[2],
            'amplitude_error': np.sqrt(pcov[2, 2]),
            'r_squared': r_squared,
            'half_life_ps': popt[0] * (-np.log(0.5))**(1/popt[1])
        }
        
        # Save to file
        import json
        with open('hbond_fitting_results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print("\nResults saved to 'hbond_fitting_results.json'")
        
        # Save fitted curve data
        fitted_data = np.column_stack((times, 
                                      stretched_exponential(times, *popt)))
        np.savetxt('fitted_curve.dat', fitted_data, 
                  header='Time(ps) C(t)_fitted', fmt='%.6f')
        
    else:
        print("Fitting failed. Try different initial guesses.")

# ============================================
# RUN THE SCRIPT
# ============================================
if __name__ == "__main__":
    main()
