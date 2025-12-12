#!/usr/bin/env python3
"""
Smart PyTorch installer that detects GPU availability and installs appropriate version.
Run this before installing requirements.txt

Usage: python install_pytorch.py
"""

import subprocess
import sys
import platform

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def get_cuda_version():
    """Get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.returncode == 0:
            # Parse CUDA version from output
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    # Extract version number
                    import re
                    match = re.search(r'CUDA Version: (\d+\.\d+)', line)
                    if match:
                        version = float(match.group(1))
                        return version
        return None
    except Exception:
        return None

def install_pytorch(use_gpu=False, cuda_version=None):
    """Install PyTorch with appropriate CUDA support."""
    
    print("=" * 60)
    print("PyTorch Installation Script")
    print("=" * 60)
    
    if use_gpu:
        print(f"\n✓ NVIDIA GPU detected!")
        print(f"✓ CUDA Version: {cuda_version}")
        
        # Determine which CUDA version to install
        if cuda_version >= 12.1:
            torch_index = "cu121"
            print("→ Installing PyTorch with CUDA 12.1 support...")
        elif cuda_version >= 11.8:
            torch_index = "cu118"
            print("→ Installing PyTorch with CUDA 11.8 support...")
        else:
            torch_index = "cu117"
            print("→ Installing PyTorch with CUDA 11.7 support...")
        
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            '--index-url', f'https://download.pytorch.org/whl/{torch_index}'
        ]
    else:
        print("\n⚠ No NVIDIA GPU detected")
        print("→ Installing CPU-only PyTorch...")
        
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch>=2.0.0',
            'torchvision>=0.15.0'
        ]
    
    # Install PyTorch
    print("\nInstalling PyTorch (this may take a few minutes)...")
    result = subprocess.run(install_cmd)
    
    if result.returncode == 0:
        print("\n✓ PyTorch installation successful!")
        
        # Verify installation
        print("\nVerifying installation...")
        verify_cmd = [
            sys.executable, '-c',
            "import torch; print(f'PyTorch: {torch.__version__}'); "
            "print(f'CUDA Available: {torch.cuda.is_available()}'); "
            "print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
        ]
        subprocess.run(verify_cmd)
        
        print("\n" + "=" * 60)
        print("Next steps:")
        print("1. Install remaining dependencies: pip install -r requirements.txt")
        print("2. Run the app: streamlit run app.py")
        print("=" * 60)
    else:
        print("\n✗ PyTorch installation failed!")
        print("Please try manual installation. See GPU_SETUP.md for help.")
        sys.exit(1)

def main():
    print("\nDetecting system configuration...\n")
    
    has_gpu = check_nvidia_gpu()
    cuda_version = get_cuda_version() if has_gpu else None
    
    print(f"Operating System: {platform.system()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"NVIDIA GPU: {'Yes' if has_gpu else 'No'}")
    
    if has_gpu and cuda_version:
        print(f"CUDA Version: {cuda_version}")
    
    print("\n" + "-" * 60)
    
    if has_gpu:
        response = input("\nGPU detected! Install GPU-accelerated PyTorch? [Y/n]: ").strip().lower()
        use_gpu = response != 'n'
    else:
        print("\nNo GPU detected. Installing CPU-only version.")
        use_gpu = False
    
    install_pytorch(use_gpu, cuda_version)

if __name__ == "__main__":
    main()