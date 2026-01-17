import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
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
    
    Supports various GROMACS formats:
    - hbond.xvg from 'gmx hbond -ac'
    - hbond.xvg from 'gmx hbond -life'
    - Any xvg file with time in first column and correlation in second
    """
    data = []
    time_unit = "ps"  # Default
    corr_label = "C(t)"
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check for unit information in comments
            if line.startswith('@'):
                if 'xaxis' in line.lower() and 'label' in line.lower():
                    if 'ns' in line.lower():
                        time_unit = "ns"
                    elif 'ps' in line.lower():
                        time_unit = "ps"
                elif 'yaxis' in line.lower() and 'label' in line.lower():
                    if 'c(' in line.lower() or 'correlation' in line.lower():
                        corr_label = "Correlation"
            
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
    
    if not data:
        print(f"ERROR: No data found in {filename}")
        print("Check if file exists and has the correct format.")
        sys.exit(1)
    
    data = np.array(data)
    times = data[:, 0]
    correlations = data[:, 1]
    
    print(f"\n{'='*60}")
    print(f"DATA SUMMARY: {os.path.basename(filename)}")
    print(f"{'='*60}")
    print(f"Data points: {len(times)}")
    print(f"Time range: {times[0]:.4f} to {times[-1]:.4f} {time_unit}")
    print(f"Time unit detected: {time_unit}")
    print(f"Correlation range: {correlations[0]:.6f} to {correlations[-1]:.6f}")
    print(f"Initial correlation: {correlations[0]:.6f}")
    
    return times, correlations, time_unit, corr_label

# ============================================
# FITTING FUNCTION
# ============================================
def fit_stretched_exponential(times, correlations, time_unit="ps", initial_guess=None):
    """
    Fit stretched exponential to correlation data.
    """
    # Default initial guess
    if initial_guess is None:
        # Estimate from data
        tau_guess = times[np.where(correlations <= correlations[0]*0.37)[0][0]] if len(np.where(correlations <= correlations[0]*0.37)[0]) > 0 else times[-1]/10
        beta_guess = 0.6  # Reasonable starting point
        A_guess = correlations[0]
        initial_guess = [tau_guess, beta_guess, A_guess]
    
    print(f"\nInitial guess: τ={initial_guess[0]:.2f} {time_unit}, "
          f"β={initial_guess[1]:.2f}, A={initial_guess[2]:.2f}")
    
    # Bounds: tau > 0, 0.1 < beta <= 1, A > 0
    bounds = ([0.001, 0.1, 0.01], [np.inf, 1.0, np.inf])
    
    try:
        popt, pcov = curve_fit(
            stretched_exponential, 
            times, 
            correlations, 
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        
        fitted_curve = stretched_exponential(times, *popt)
        
        # Calculate R-squared
        residuals = correlations - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((correlations - np.mean(correlations))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1
        
        # Calculate standard errors
        perr = np.sqrt(np.diag(pcov))
        
        print(f"\n{'='*60}")
        print("FITTING RESULTS")
        print(f"{'='*60}")
        print(f"τ (relaxation time) = {popt[0]:.6f} ± {perr[0]:.6f} {time_unit}")
        print(f"β (stretching exponent) = {popt[1]:.6f} ± {perr[1]:.6f}")
        print(f"A (amplitude) = {popt[2]:.6f} ± {perr[2]:.6f}")
        print(f"R² = {r_squared:.6f}")
        
        # Calculate half-life and 1/e lifetime
        t_half = popt[0] * (-np.log(0.5))**(1/popt[1])
        t_1e = popt[0] * (-np.log(1/np.e))**(1/popt[1])  # Time to decay to 1/e
        print(f"Half-life (C(t)=0.5*A): {t_half:.6f} {time_unit}")
        print(f"1/e lifetime (C(t)=A/e): {t_1e:.6f} {time_unit}")
        print(f"{'='*60}")
        
        return popt, pcov, fitted_curve, r_squared
        
    except Exception as e:
        print(f"\nERROR: Fitting failed: {e}")
        print("Try providing initial guesses with --tau, --beta, --amp options")
        return None, None, None, None

# ============================================
# PLOTTING
# ============================================
def create_plots(times, correlations, fitted_curve, popt, 
                 time_unit="ps", corr_label="C(t)", filename="hbond"):
    """
    Create comprehensive plots.
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Linear scale plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(times, correlations, alpha=0.5, label='Data', s=10, color='blue')
    ax1.plot(times, fitted_curve, 'r-', linewidth=2, label='Stretched Exp Fit')
    ax1.set_xlabel(f'Time ({time_unit})', fontsize=11)
    ax1.set_ylabel(corr_label, fontsize=11)
    ax1.set_title('Linear Scale', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add fitting info
    info_text = f'τ = {popt[0]:.4f} {time_unit}\nβ = {popt[1]:.4f}\nA = {popt[2]:.4f}'
    ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Log-log scale
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(times, correlations, alpha=0.5, s=10, color='blue')
    ax2.plot(times, fitted_curve, 'r-', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(f'Time ({time_unit}, log)', fontsize=11)
    ax2.set_ylabel(f'{corr_label} (log)', fontsize=11)
    ax2.set_title('Log-Log Scale', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Semi-log (log time)
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(times, correlations, alpha=0.5, s=10, color='blue')
    ax3.plot(times, fitted_curve, 'r-', linewidth=2)
    ax3.set_xscale('log')
    ax3.set_xlabel(f'Time ({time_unit}, log)', fontsize=11)
    ax3.set_ylabel(corr_label, fontsize=11)
    ax3.set_title('Semi-Log (log time)', fontsize=12)
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Semi-log (log correlation)
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(times, correlations, alpha=0.5, s=10, color='blue')
    ax4.plot(times, fitted_curve, 'r-', linewidth=2)
    ax4.set_yscale('log')
    ax4.set_xlabel(f'Time ({time_unit})', fontsize=11)
    ax4.set_ylabel(f'{corr_label} (log)', fontsize=11)
    ax4.set_title('Semi-Log (log correlation)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuals plot
    residuals = correlations - fitted_curve
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(times, residuals, alpha=0.5, s=10, color='green')
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax5.set_xlabel(f'Time ({time_unit})', fontsize=11)
    ax5.set_ylabel('Residuals', fontsize=11)
    ax5.set_title(f'Residuals (Mean: {np.mean(residuals):.2e})', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 6. Histogram of residuals
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax6.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Residuals', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title(f'Residuals Distribution\nStd: {np.std(residuals):.2e}', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Hydrogen Bond Lifetime Correlation: {filename}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plot_filename = f"{filename}_fit.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    return fig

# ============================================
# SAVE RESULTS
# ============================================
def save_results(times, correlations, fitted_curve, popt, perr, r_squared, 
                 time_unit, filename="hbond"):
    """
    Save all results to files.
    """
    import json
    from datetime import datetime
    
    # 1. Save fitting parameters
    results = {
        'fitting_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_file': filename,
        'tau': float(popt[0]),
        'tau_error': float(perr[0]),
        'tau_unit': time_unit,
        'beta': float(popt[1]),
        'beta_error': float(perr[1]),
        'amplitude': float(popt[2]),
        'amplitude_error': float(perr[2]),
        'r_squared': float(r_squared),
        'half_life': float(popt[0] * (-np.log(0.5))**(1/popt[1])),
        'half_life_unit': time_unit,
        'e_folding_time': float(popt[0] * (-np.log(1/np.e))**(1/popt[1])),
        'e_folding_unit': time_unit,
        'data_points': int(len(times))
    }
    
    with open(f'{filename}_fit_params.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 2. Save fitted curve data
    fitted_data = np.column_stack((times, correlations, fitted_curve, 
                                  correlations - fitted_curve))
    np.savetxt(f'{filename}_fitted_data.dat', fitted_data,
              header=f'Time({time_unit}) Original_C(t) Fitted_C(t) Residuals',
              fmt='%.8f')
    
    # 3. Save summary to text file
    with open(f'{filename}_summary.txt', 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"HYDROGEN BOND LIFETIME FITTING RESULTS\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Input file: {filename}\n")
        f.write(f"Date: {results['fitting_date']}\n\n")
        f.write(f"{'='*60}\n")
        f.write("FITTING PARAMETERS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Relaxation time (τ): {popt[0]:.6f} ± {perr[0]:.6f} {time_unit}\n")
        f.write(f"Stretching exponent (β): {popt[1]:.6f} ± {perr[1]:.6f}\n")
        f.write(f"Amplitude (A): {popt[2]:.6f} ± {perr[2]:.6f}\n")
        f.write(f"R-squared: {r_squared:.6f}\n\n")
        f.write(f"Half-life (C(t)=0.5): {results['half_life']:.6f} {time_unit}\n")
        f.write(f"1/e lifetime: {results['e_folding_time']:.6f} {time_unit}\n")
        f.write(f"{'='*60}\n")
    
    print(f"\nResults saved to:")
    print(f"  - {filename}_fit_params.json (JSON format)")
    print(f"  - {filename}_fitted_data.dat (numerical data)")
    print(f"  - {filename}_summary.txt (text summary)")

# ============================================
# COMMAND LINE INTERFACE
# ============================================
def parse_arguments():
    """
    Parse command line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fit stretched exponential to GROMACS hydrogen bond lifetime data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s hbond.xvg
  %(prog)s hbond.xvg --tau 100 --beta 0.7
  %(prog)s hbond.xvg --min-time 1.0 --max-time 1000
  %(prog)s hbond.xvg --output myfit
        """
    )
    
    parser.add_argument('filename', help='GROMACS XVG file with correlation data')
    parser.add_argument('--tau', type=float, help='Initial guess for τ (relaxation time)')
    parser.add_argument('--beta', type=float, help='Initial guess for β (stretching exponent)')
    parser.add_argument('--amp', type=float, help='Initial guess for A (amplitude)')
    parser.add_argument('--min-time', type=float, help='Minimum time to include in fit')
    parser.add_argument('--max-time', type=float, help='Maximum time to include in fit')
    parser.add_argument('--output', type=str, default='', help='Base name for output files')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    return parser.parse_args()

# ============================================
# MAIN FUNCTION
# ============================================
def main():
    """
    Main function for command line execution.
    """
    args = parse_arguments()
    
    # Check if file exists
    if not os.path.exists(args.filename):
        print(f"ERROR: File '{args.filename}' not found!")
        sys.exit(1)
    
    # Load data
    print(f"\nLoading data from: {args.filename}")
    times, correlations, time_unit, corr_label = load_gromacs_hbond_correlation(args.filename)
    
    # Apply time range filters if specified
    if args.min_time is not None or args.max_time is not None:
        mask = np.ones_like(times, dtype=bool)
        if args.min_time is not None:
            mask = mask & (times >= args.min_time)
            print(f"Applying minimum time cutoff: {args.min_time} {time_unit}")
        if args.max_time is not None:
            mask = mask & (times <= args.max_time)
            print(f"Applying maximum time cutoff: {args.max_time} {time_unit}")
        
        times = times[mask]
        correlations = correlations[mask]
        print(f"Using {len(times)} data points after time filtering")
    
    # Set initial guess
    initial_guess = None
    if args.tau or args.beta or args.amp:
        tau_guess = args.tau if args.tau else times[-1]/10
        beta_guess = args.beta if args.beta else 0.6
        amp_guess = args.amp if args.amp else correlations[0]
        initial_guess = [tau_guess, beta_guess, amp_guess]
        print(f"\nUser-provided initial guess: τ={tau_guess}, β={beta_guess}, A={amp_guess}")
    
    # Perform fitting
    print(f"\n{'='*60}")
    print("FITTING STRETCHED EXPONENTIAL")
    print(f"{'='*60}")
    
    result = fit_stretched_exponential(times, correlations, time_unit, initial_guess)
    
    if result[0] is None:
        print("Fitting failed. Exiting.")
        sys.exit(1)
    
    popt, pcov, fitted_curve, r_squared = result
    perr = np.sqrt(np.diag(pcov))
    
    # Determine output base name
    if args.output:
        base_name = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.filename))[0]
    
    # Create plots if not disabled
    if not args.no_plot:
        print("\nGenerating plots...")
        fig = create_plots(times, correlations, fitted_curve, popt, 
                          time_unit, corr_label, base_name)
        plt.show()
    else:
        print("\nPlotting disabled (--no-plot flag used)")
    
    # Save results
    save_results(times, correlations, fitted_curve, popt, perr, r_squared, 
                 time_unit, base_name)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

# ============================================
# EXECUTE
# ============================================
if __name__ == "__main__":
    main()
